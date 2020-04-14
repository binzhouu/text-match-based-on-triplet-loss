# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-02-20
"""
import os
import sys
root_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
sys.path.append(root_path)
from constants import ROOT_PATH
import pickle
import torch
import numpy as np
import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
from models.text_match_v1.utils.data import Data
from models.text_match_v1.model.ccnn_wlstm import TextMatchModel
from models.text_match_v1.utils.functions import normalize_word, predict_batchfy_classification_with_label, get_represents

dset_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/alphabet.dset')
represent_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/represent.dset')
index_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/large.index')
no_train_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/add/add.csv')
model_dir = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/text_match_v1.model')
user_dict = os.path.join(ROOT_PATH, 'lexicon/user_dict.txt')

logger = logging.getLogger(__name__)


class SceneMatch(object):
	def __init__(self):
		self.data = Data(ori_texts=[], labels=[], if_train=False)
		with open(dset_path, 'rb') as rbf:
			self.data.char_alphabet.instance2index = pickle.load(rbf)
			self.data.word_alphabet.instance2index = pickle.load(rbf)
			self.data.label_alphabet.instance2index = pickle.load(rbf)
			self.data.char_alphabet_size = pickle.load(rbf)
			self.data.word_alphabet_size = pickle.load(rbf)
			self.data.label_alphabet_size = pickle.load(rbf)
			self.data.label_alphabet.instances = pickle.load(rbf)
		self.data.fix_alphabet()
		self.model = TextMatchModel(self.data)
		self.model.load_state_dict(torch.load(model_dir, map_location=self.model.configs['map_location']))
		self.model.eval()
		self.model.to(self.model.configs['device'])
		# self.no_train_texts, self.no_train_represents, self.no_train_label_ids = get_represents(
		# 	self.data, self.model, 'add', self.model.configs)

	def inference(self, text, text_list, label_list):
		texts, ids = self.data.read_scene_text_list(text_list, label_list)
		self.data.scene_texts, self.data.scene_ids = texts, ids
		self.scene_texts, scene_represents, scene_label_ids = get_represents(self.data, self.model, 'scene', self.model.configs)
		# 处理当前传入的用户input_text
		texts, ids = [], []
		seg_list = self.data.segment([text])[0]
		if len(seg_list) == 0:
			return None, None, None
		# print('seg_list: %s' % seg_list)
		char_list, char_id_list, word_id_list, = [], [], []
		for word in seg_list:
			word_id = self.data.word_alphabet.get_index(normalize_word(word))
			word_id_list.append(word_id)
			chars, char_ids = [], []
			if self.data.specific_word(word):
				chars.append(word)
				char_ids.append(self.data.char_alphabet.get_index(normalize_word(word)))
			else:
				for char in word:
					chars.append(char)
					char_ids.append(self.data.char_alphabet.get_index(normalize_word(char)))
			char_list.append(chars)
			char_id_list.append(char_ids)
		texts.append([seg_list, char_list])
		ids.append([word_id_list, char_id_list])
		batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, mask = \
			predict_batchfy_classification_with_label(ids, self.model.configs['gpu'], if_train=False)
		pred_represent = self.model(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
		max_score, max_similar_text = self.cal_similarity(pred_represent, scene_represents)
		pred_text = ''.join(max_similar_text[0])
		pred_label = max_similar_text[-1]
		if pred_label == 'None':
			pred_label = None
		# 置信度、最接近的text，最接近的label
		return max_score, pred_text, pred_label

	def cal_similarity(self, pred_represent, train_represents):
		pred_represent = pred_represent.cpu().data.numpy()
		score = cosine_similarity(pred_represent, train_represents)
		max_id, max_score = np.argmax(score, axis=-1)[0], np.max(score, axis=-1)[0]
		# max_simialr_text, max_similar_label_id = self.train_texts[max_id], self.train_label_ids[max_id]
		max_simialr_text = self.scene_texts[max_id]
		return max_score, max_simialr_text


if __name__ == '__main__':
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
	start = datetime.datetime.now()
	sm = SceneMatch()
	print('model inits time costs: %s' % (datetime.datetime.now() - start).total_seconds())
	for text in texts:
		start = datetime.datetime.now()
		confidence, similar_text, pred_label = sm.inference(text, text_list, label_list)
		end = datetime.datetime.now()
		print('text:%s, pred_label:%s, confidence:%s, similar_text: %s,  time costs: %s' % (
			text, pred_label, confidence, similar_text, (end - start).total_seconds()))
