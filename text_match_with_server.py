# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-02-27
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
# import faiss
import grpc
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from models.text_match_v1.utils.data import Data
from models.text_match_v1.model.ccnn_wlstm import TextMatchModel
from models.text_match_v1.utils.functions import normalize_word, predict_batchfy_classification_with_label, \
	get_represents, build_pretrain_embedding
from models.text_match_v1.protos.faiss_server_pb2_grpc import FaissServerStub
from models.text_match_v1.protos.faiss_server_pb2 import FloatVector, SearchRequest
import traceback

dset_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/alphabet.dset')
represent_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/represent.dset')
index_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/large.index')
no_train_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/add/no_train.csv')
model_dir = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/text_match_v1.model')
user_dict = os.path.join(ROOT_PATH, 'lexicon/user_dict.txt')
glove_emb_path = os.path.join(ROOT_PATH, 'models/glove_emb/glove_ch_vec.pkl')

logger = logging.getLogger(__name__)


class TextMatch(object):
	def __init__(self, ip_port, if_write=False):
		"""
		:param ip_port: faiss url
		:param if_write:
		"""
		self.data = Data(ori_texts=[], labels=[], if_train=False)
		with open(dset_path, 'rb') as rbf:
			self.data.char_alphabet.instance2index = pickle.load(rbf)
			self.data.word_alphabet.instance2index = pickle.load(rbf)
			self.data.label_alphabet.instance2index = pickle.load(rbf)
			self.data.char_alphabet_size = pickle.load(rbf)
			self.data.word_alphabet_size = pickle.load(rbf)
			self.data.label_alphabet_size = pickle.load(rbf)
			self.data.label_alphabet.instances = pickle.load(rbf)
			self.data.train_texts = pickle.load(rbf)
		self.data.fix_alphabet()
		self.model = TextMatchModel(self.data)
		self.model.load_state_dict(torch.load(model_dir, map_location=self.model.configs['map_location']))
		self.model.eval()
		self.model.to(self.model.configs['device'])
		self.train_represents = np.zeros(shape=())
		self.train_texts = self.data.train_texts
		# 场景匹配的初始化
		self.scene_texts = []
		self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(glove_emb_path, self.data.char_alphabet)
		self.char_embedding = nn.Embedding(self.data.char_alphabet_size, self.char_emb_dim)
		self.char_embedding.weight.data.copy_(torch.from_numpy(self.pretrain_char_embedding))
		if self.model.configs['gpu']:
			self.char_embedding = self.char_embedding.cuda()
		# if if_write:
		# 	self.write_index()
		self.channel = grpc.insecure_channel(ip_port)
		self.stub = FaissServerStub(self.channel)

	def close(self):
		self.channel.close()

	# def write_index(self):
	# 	"""
	# 	余弦和欧式距离等价：
	# 	https://www.zhihu.com/question/19640394
	# 	:return:
	# 	"""
	# 	import faiss
	# 	with open(represent_path, 'rb') as rbf:
	# 		self.train_represents = pickle.load(rbf)
	# 	self.train_represents = np.array(self.train_represents).astype('float32')
	# 	d = self.model.configs['num_output']
	# 	nlist = self.data.label_alphabet_size - 1
	# 	# L2距离计算方式
	# 	# quantizer = faiss.IndexFlatL2(d)
	# 	# index = faiss.IndexIVFFlat(quantizer, d, nlist)
	# 	# 余弦计算方式
	# 	quantizer = faiss.IndexFlatIP(d)
	# 	index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
	# 	faiss.normalize_L2(self.train_represents)  # 规一化
	# 	# print(index.is_trained)
	# 	index.train(self.train_represents)
	# 	index.add_with_ids(self.train_represents, np.arange(self.train_represents.shape[0]))
	# 	index.nprobe = 10
	# 	faiss.write_index(index, index_path)

	def inference(self, text):
		"""

		:param text:
		:return:
		"""
		texts, ids = [], []
		seg_list = self.data.segment([text])[0]
		if len(seg_list) == 0:
			return 1, None, None
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
		pred_represent = pred_represent.data.numpy()
		# ori_pred_represent = pred_represent
		# faiss.normalize_L2(pred_represent)
		# numpy改写faiss.normalize_L2
		pred_represent = pred_represent / np.linalg.norm(pred_represent, ord=2)
		pred_represent = pred_represent.tolist()[0]
		D, I = self.search(self.stub, pred_represent)
		if D > 0 and I > 0:
			max_id = I[0][0]
			max_score = D
			max_similar_text = self.train_texts[max_id]
			pred_text = ''.join(max_similar_text[0])
			pred_label = max_similar_text[-1]
			if pred_label == 'None':
				pred_label = None
			return max_score, pred_text, pred_label
		else:
			# 如果faiss调用失败，返回默认得分和标签
			return 0, None, None

	def inference_for_scene(self, text, text_list, label_list):
		"""
		for scene match
		:param text:
		:param text_list:
		:param label_list:
		:return:

		"""
		texts, ids = self.data.read_scene_text_list(text_list, label_list)
		self.data.scene_texts, self.data.scene_ids = texts, ids
		self.scene_texts, scene_represents, scene_label_ids = get_represents(
			self.data, self.model, 'scene', self.model.configs)
		# 处理当前传入的用户input_text
		texts, ids = [], []
		seg_list = self.data.segment([text])[0]
		if len(seg_list) == 0:
			return 1, None, None
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

	def inference_for_scene_with_glove(self, text, text_list, label_list):
		# 计算scene_texts的表征
		_, scene_ids = self.data.read_scene_text_list(text_list, label_list)
		char_inputs = []
		for word_ids, char_ids, label_ids in scene_ids:
			new_char_id = []
			for ii in char_ids:
				for i in ii:
					new_char_id.append(i)
			char_inputs.append(new_char_id)
		scene_represents = []
		for char_input in char_inputs:
			char_input = torch.tensor(char_input, dtype=torch.long).to(self.model.configs['device'])
			char_embedding = self.char_embedding(char_input)
			sentence_embedding = torch.mean(char_embedding, dim=0)
			sentence_embedding = sentence_embedding.cpu().data.numpy().tolist()
			scene_represents.append(sentence_embedding)
		# 处理当前input_text:
		texts, ids = [], []
		seg_list = self.data.segment([text])[0]
		if len(seg_list) == 0:
			return 1, None, None
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
		pred_char_inputs = []
		char_ids = ids[0][-1]
		for ii in char_ids:
			for i in ii:
				pred_char_inputs.append(i)
		pred_inputs = torch.tensor(pred_char_inputs, dtype=torch.long).to(self.model.configs['device'])
		pred_represent = self.char_embedding(pred_inputs)
		pred_represent = torch.mean(pred_represent, dim=0).unsqueeze(0)
		max_score, pred_text, pred_label = self.cal_similarity(pred_represent, scene_represents, text_list, label_list)
		if pred_label == 'None':
			pred_label = None
		# 置信度、最接近的text，最接近的label
		return max_score, pred_text, pred_label

	def cal_similarity(self, pred_represent, train_represents, text_list, label_list):
		pred_represent = pred_represent.cpu().data.numpy()
		score = cosine_similarity(pred_represent, train_represents)
		max_id, max_score = np.argmax(score, axis=-1)[0], np.max(score, axis=-1)[0]
		max_simialr_text, max_similar_label = text_list[max_id], label_list[max_id]
		return max_score, max_simialr_text, max_similar_label

	@staticmethod
	def search(stub, input_vector, topn=1, index_name='text_match'):
		float_vector = FloatVector()
		for i in input_vector:
			float_vector.fvec.append(i)
		search_request = SearchRequest(index_name=index_name, vector=float_vector, topn=topn)
		try:
			response = stub.search(search_request, timeout=0.1)
		# get faiss服务异常：
		except Exception as exc:
			logger.error("Respose failed: {}".format(traceback.format_exc()))  # format_exc:将异常信息记录在log里
			return -1, -1
		if response.success:
			_D = response.D
			D = []
			for v in _D.fmatrix:
				d = []
				for e in v.fvec:
					d.append(e)
				D.append(d)
			D = np.array(D)
			_I = response.I
			I = []
			for v in _I.imatrix:
				i = []
				for e in v.ivec:
					i.append(e)
				I.append(i)
			I = np.array(I)
			if isinstance(D, np.ndarray):
				D = D[0][0]
			return D, I
		else:
			# 如果faiss调用失败，返回默认值
			logger.error("Faiss server failed, Return default value.")
			return -1, -1


if __name__ == '__main__':
	# texts = [
	# 	'我回家了', 'M', '2', '110', '我不知道啊', '我查YOUYOU没出来啊', 'U', 'V', ' ', '你不用不用你唤醒了', '你能干什么', '卧室', '空调', '设备',
	# 	'<', '>', '&', '*', '?', '什么', '咋了', '给你起个名字叫小狗子吧', '小安', '安', '安全', '查', '加', '湿', '打', '窗帘',
	# 	'制冷', '制热', '打开蓝牙', '灯光', '是的', '声音大一点', '声音大一些', '设置温度为20', '调节温度到20', '晾衣架调高',
	# 	'晾衣架高度调高', '客厅', '打开加热', '关闭加热功能', '我不想回家啊', '查打开', 'test']
	# # texts = [
	# # 	'打开灯', '打开电视', '把音量调高一点', '空调的湿度调低', '扫地机回去充电', '扫地机设为回充模式', '晾衣架调高一点', '客厅的灯调亮一点', '风扇调小一点',
	# # 	'打开空调', '打开豆浆机', '我回来了', '回来啦', '温度调高2度', '湿度调高2度', '打开香薰机的灯', '关闭台灯的炫彩模式', '关闭空调的自动模式', '关闭静音模式',
	# # 	'暂停播放', '暂停一下', '暂停电视机', '暂停房间的空调']
	# # texts = ['干啥呢']
	# start = datetime.datetime.now()
	# # tm = TextMatch(ip_port='localhost:50051', if_write=True)
	# tm = TextMatch(ip_port='localhost:50051')
	# print('model inits time costs: %s' % (datetime.datetime.now() - start).total_seconds())
	# for text in texts:
	# 	start = datetime.datetime.now()
	# 	confidence, similar_text, pred_label = tm.inference(text)
	# 	end = datetime.datetime.now()
	# 	print('text:%s, pred_label:%s, confidence:%s, similar_text: %s,  time costs: %s' % (
	# 		text, pred_label, confidence, similar_text, (end - start).total_seconds()))
	# tm.close()

	tm = TextMatch(ip_port='localhost:50051')
	scene_name = []
	text_list, label_list = [], []
	scenes = [{'name': '回家'}, {'name': '休息'}, {'name': '出门'}, {'name': '房间灯'}]
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
	texts = [
		'打开房间的灯出门吧', '我回家了啊', '开启回家', '打开回家吧', '开回家场景', '打开休息模式吧', '吃饭吧', '开门', '我下班了', '打开灯', '呵呵', '我不想睡觉呀',
		'我出门了', 'test', '开房间灯', '打开', '平时我打开了']
	for text in texts:
		start_time = datetime.datetime.now()
		confidence, similar_text, pred_label = tm.inference_for_scene_with_glove(text, text_list, label_list)
		end_time = datetime.datetime.now()
		# confidence, similar_text, pred_label = tm.inference_for_scene(text, text_list, label_list)
		print('input_text: %s, conf: %s, similar_text: %s, pred_label: %s, time costs: %s' % (
			text, confidence, similar_text, pred_label, (end_time-start_time).total_seconds()))
