# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-02-16
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
from pkuseg import pkuseg
import logging
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from models.text_match_v1.utils.data import Data
from models.text_match_v1.model.ccnn_wlstm import TextMatchModel
from models.text_match_v1.utils.functions import normalize_word, predict_batchfy_classification_with_label, get_represents

dset_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/alphabet.dset')
represent_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/represent.dset')
index_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/large.index')
no_train_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/add/no_train.csv')
model_dir = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/text_match_v1.model')
user_dict = os.path.join(ROOT_PATH, 'lexicon/user_dict.txt')

logger = logging.getLogger(__name__)


class TextMatch(object):
	def __init__(self, if_write=False):
		self.data = Data(ori_texts=[], labels=[], if_train=False)
		with open(dset_path, 'rb') as rbf:
			self.data.char_alphabet.instance2index = pickle.load(rbf)
			self.data.word_alphabet.instance2index = pickle.load(rbf)
			self.data.label_alphabet.instance2index = pickle.load(rbf)
			self.data.char_alphabet_size = pickle.load(rbf)
			self.data.word_alphabet_size = pickle.load(rbf)
			self.data.label_alphabet_size = pickle.load(rbf)
			self.data.label_alphabet.instances = pickle.load(rbf)
			# s = datetime.datetime.now()
			self.data.train_texts = pickle.load(rbf)
			# self.data.train_ids = pickle.load(rbf)
			# run save_train_represents.py
			# self.train_texts = pickle.load(rbf)
			# self.train_represents = pickle.load(rbf)
			# self.train_label_ids = pickle.load(rbf)
			# print('costs: %s' % (datetime.datetime.now()-s).total_seconds())
		self.data.fix_alphabet()
		self.model = TextMatchModel(self.data)
		self.model.load_state_dict(torch.load(model_dir, map_location=self.model.configs['map_location']))
		self.model.eval()
		self.model.to(self.model.configs['device'])
		self.data.no_train_texts, self.data.no_train_ids = self.data.read_no_train(no_train_path)
		self.no_train_texts, self.no_train_represents, self.no_train_label_ids = get_represents(
			self.data, self.model, 'no_train', self.model.configs)
		self.train_represents = np.zeros(shape=())
		self.train_texts = self.data.train_texts + self.no_train_texts
		if if_write:
			self.write_index()

	def write_index(self):
		"""
		余弦和欧式距离等价：
		https://www.zhihu.com/question/19640394
		:return:
		"""
		with open(represent_path, 'rb') as rbf:
			self.train_represents = pickle.load(rbf)
		self.train_represents = np.array(self.train_represents).astype('float32')
		d = self.model.configs['num_output']
		nlist = self.data.label_alphabet_size - 1
		# L2距离计算方式
		# quantizer = faiss.IndexFlatL2(d)
		# index = faiss.IndexIVFFlat(quantizer, d, nlist)
		# 余弦计算方式
		quantizer = faiss.IndexFlatIP(d)
		index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
		faiss.normalize_L2(self.train_represents)  # 规一化
		# print(index.is_trained)
		index.train(self.train_represents)
		index.add_with_ids(self.train_represents, np.arange(self.train_represents.shape[0]))
		index.nprobe = 10
		faiss.write_index(index, index_path)

	def load_index(self, add_with_ids=False):
		try:
			index = faiss.read_index(index_path)
			if add_with_ids:
				# 添加额外的样本
				if not isinstance(self.no_train_represents, np.ndarray):
					self.no_train_represents = np.array(self.no_train_represents).astype('float32')
				faiss.normalize_L2(self.no_train_represents)  # 归一化
				ids = np.arange(index.ntotal, index.ntotal + len(self.no_train_represents))
				index.add_with_ids(self.no_train_represents, ids)
			return index
		except FileNotFoundError:
			logger.info("index file does not exist !")

	def inferecne(self, text, index):
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
		pred_represent = pred_represent.data.numpy()
		faiss.normalize_L2(pred_represent)  # 归一化
		D, I = index.search(pred_represent, 1)
		max_id = I[0][0]
		max_score = D  # ??
		max_similar_text = self.train_texts[max_id]
		pred_text = ''.join(max_similar_text[0])
		pred_label = max_similar_text[-1]
		return max_score, pred_text, pred_label


if __name__ == '__main__':
	texts = [
		'我回家了', 'M', '2', '110', '我不知道啊', '我查YOUYOU没出来啊', 'U', 'V', ' ', '你不用不用你唤醒了', '你能干什么', '卧室', '空调', '设备',
		'<', '>', '&', '*', '?', '什么', '咋了', '给你起个名字叫小狗子吧', '小安', '安', '安全', '查', '加', '湿', '打', '窗帘',
		'制冷', '制热', '打开蓝牙', '灯光', '是的', '声音大一点', '声音大一些', '设置温度为20', '调节温度到20', '晾衣架调高',
		'晾衣架高度调高', '客厅', '打开加热', '关闭加热功能', '我不想回家啊', '查打开']
	# texts = [
	# 	'打开灯', '打开电视', '把音量调高一点', '空调的湿度调低', '扫地机回去充电', '扫地机设为回充模式', '晾衣架调高一点', '客厅的灯调亮一点', '风扇调小一点',
	# 	'打开空调', '打开豆浆机', '我回来了', '回来啦', '温度调高2度', '湿度调高2度', '打开香薰机的灯', '关闭台灯的炫彩模式', '关闭空调的自动模式', '关闭静音模式',
	# 	'暂停播放', '暂停一下', '暂停电视机', '暂停房间的空调']
	# texts = ['我不想回家啊', '那人是个警察啊', '呵呵呵']
	start = datetime.datetime.now()
	# tm = TextMatch(if_write=True)
	tm = TextMatch(if_write=False)
	# index = tm.load_index(add_with_ids=True)
	index = tm.load_index(add_with_ids=False)
	print('model inits time costs: %s' % (datetime.datetime.now() - start).total_seconds())
	for text in texts:
		start = datetime.datetime.now()
		confidence, similar_text, pred_label = tm.inferecne(text, index)
		end = datetime.datetime.now()
		print('text:%s, pred_label:%s, confidence:%s, similar_text: %s,  time costs: %s' % (
			text, pred_label, confidence, similar_text, (end - start).total_seconds()))
