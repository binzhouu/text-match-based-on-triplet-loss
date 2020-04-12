# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-01-17
"""
import torch
import torch.nn as nn
import os
import yaml
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from constants import ROOT_PATH
from models.text_match_v1.model.triplet_loss import TripletLoss
from models.text_match_v1.utils.functions import get_represents, normalize_word, predict_batchfy_classification_with_label
from models.text_match_v1.utils.data import Data

model_yml_path = os.path.join(ROOT_PATH, 'models/text_match_v1/conf/model.yml')
with open(model_yml_path, 'r') as rf:
	model_configs = yaml.load(rf, Loader=yaml.FullLoader)


class TextMatchModel(nn.Module):
	def __init__(self, data):
		super(TextMatchModel, self).__init__()
		self.configs = self.read_configs()
		self.loss = TripletLoss()
		self.data = data

		if self.configs['random_embedding']:
			self.char_embeddings = nn.Embedding(data.char_alphabet_size, self.configs['char_emb_dim'])
			self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
				data.char_alphabet_size, self.configs['char_emb_dim'])))
			self.char_drop = nn.Dropout(self.configs['dropout'])

			self.word_embeddings = nn.Embedding(data.word_alphabet_size, self.configs['word_emb_dim'])
			self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
				data.word_alphabet_size, self.configs['word_emb_dim'])))
			self.word_drop = nn.Dropout(self.configs['dropout'])
		else:
			pass
		self.char_cnn = nn.Conv1d(
			in_channels=self.configs['char_emb_dim'], out_channels=self.configs['char_hidden_dim'],
			kernel_size=self.configs['kernel_size'], padding=self.configs['padding'])
		self.lstm = nn.LSTM(
			self.configs['char_hidden_dim'] + self.configs['word_emb_dim'], self.configs['word_hidden_dim']//2,
			num_layers=self.configs['num_layers'], batch_first=True, bidirectional=True)
		self.drop_lstm = nn.Dropout(self.configs['dropout'])
		# data.label_alphabet_size大小比label数量大1，是合理的，与label_alphabet的初始化策略有关
		# data.train_ids中，没有一个label值是0，所以softmax_logits[0]也一定是一个非常小的值，取不到
		# self.hidden2tag = nn.Linear(self.configs['word_hidden_dim'], data.label_alphabet_size)
		self.fc = nn.Linear(self.configs['word_hidden_dim'], self.configs['num_output'])

	def forward(self, batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, batch_label=None):
		char_batch_size = batch_char.size(0)
		char_embeds = self.char_drop(self.char_embeddings(batch_char)).transpose(1, 2)
		char_cnn_out = self.char_cnn(char_embeds)
		char_cnn_out = torch.max_pool1d(char_cnn_out, kernel_size=char_cnn_out.size(2))
		char_cnn_out = char_cnn_out.view(char_batch_size, -1)
		char_cnn_out = char_cnn_out[batch_charrecover]
		char_features = char_cnn_out.view(batch_word.size(0), batch_word.size(1), -1)

		word_embeds = self.word_embeddings(batch_word)
		word_embeds = torch.cat([word_embeds, char_features], 2)
		word_represent = self.word_drop(word_embeds)

		packed_words = pack_padded_sequence(word_represent, batch_wordlen.cpu().numpy(), batch_first=True)
		hidden = None
		lstm_out, hidden = self.lstm(packed_words, hidden)

		batch_size = batch_word.size(0)
		hidden_outputs = hidden[0].transpose(1, 0).contiguous().view(batch_size, -1)  # (64, 200)
		outputs = self.word_drop(hidden_outputs)
		logits = self.fc(outputs)  # 句子表征

		if batch_label is not None:
			loss = self.loss(batch_label, logits)
			return loss, logits
		else:
			return logits

	@staticmethod
	def read_configs():
		configs = model_configs
		# 读取设备基本属性
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		gpu = True if device.type == 'cuda' else False
		map_location = 'cpu' if gpu is False else None
		configs.update({'device': device, 'gpu': gpu, 'map_location': map_location})
		# 读取model_num对应的模型超参数
		model_num = configs['model_num']
		for k, v in configs['model'][model_num].items():
			configs[k] = v
		del configs['model']
		return configs

	@staticmethod
	def random_embedding(vocab_size, embedding_dim):
		pretrain_emb = np.empty([vocab_size, embedding_dim])
		scale = np.sqrt(3.0 / embedding_dim)
		for index in range(vocab_size):
			pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
		return pretrain_emb


if __name__ == '__main__':
	# 场景匹配的demo：
	dset_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/alphabet.dset')
	model_dir = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/text_match_v1.model')
	data = Data(ori_texts=[], labels=[], if_train=False)
	with open(dset_path, 'rb') as rbf:
		data.char_alphabet.instance2index = pickle.load(rbf)
		data.word_alphabet.instance2index = pickle.load(rbf)
		data.label_alphabet.instance2index = pickle.load(rbf)
		data.char_alphabet_size = pickle.load(rbf)
		data.word_alphabet_size = pickle.load(rbf)
		data.label_alphabet_size = pickle.load(rbf)
		data.label_alphabet.instances = pickle.load(rbf)
		data.train_texts = pickle.load(rbf)
	data.fix_alphabet()
	model = TextMatchModel(data)
	model.load_state_dict(torch.load(model_dir, map_location=model.configs['map_location']))
	model.eval()
	model.to(model.configs['device'])
	# 准备场景的测试语料
	scene_name = []
	text_list, label_list = [], []
	scenes = [{'name': '回家'}, {'name': '休息'}]
	for scene in scenes:
		scene_name.append(scene['name'])
		text_list.append(scene['name'])  # add scene name
		label_list.append('start_scene')
		# 扩展话术:
		text_list.append('打开' + scene['name'] + '场景')  # add scene name
		label_list.append('start_scene')
		text_list.append('开启' + scene['name'] + '场景')  # add scene name
		label_list.append('start_scene')
		text_list.append('打开' + scene['name'] + '模式')  # add scene name
		label_list.append('start_scene')
		text_list.append('开启' + scene['name'] + '模式')  # add scene name
		label_list.append('start_scene')
		text_list.append('关闭' + scene['name'] + '模式')
		label_list.append('close_scene')
	texts = ['我回家了啊', '关闭回家把', '开启回家', '打开回家吧', '开回家场景', '打开休息模式吧']
	for text in texts:
		confidence, similar_text, pred_label = model.inference_for_scene(text, text_list, label_list, model)
		print(confidence, similar_text, pred_label)
