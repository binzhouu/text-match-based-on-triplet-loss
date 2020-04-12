# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-02-09
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
from sklearn.metrics.pairwise import cosine_similarity
from models.text_match_v1.utils.data import Data
from models.text_match_v1.model.ccnn_wlstm import TextMatchModel
from models.text_match_v1.utils.functions import normalize_word, predict_batchfy_classification_with_label, get_represents

dset_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/alphabet.dset')
represent_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/represent.dset')
no_train_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/add/no_train.csv')
model_dir = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/text_match_v1.model')
user_dict = os.path.join(ROOT_PATH, 'lexicon/user_dict.txt')

logger = logging.getLogger(__name__)


class TextMatch(object):
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
			s = datetime.datetime.now()
			self.data.train_texts = pickle.load(rbf)
			# self.data.train_ids = pickle.load(rbf)
			# run save_train_represents.py
			# self.train_texts = pickle.load(rbf)
			with open(represent_path, 'rb') as rbf:
				self.train_represents = pickle.load(rbf)
			# self.train_label_ids = pickle.load(rbf)
			print('costs: %s' % (datetime.datetime.now()-s).total_seconds())
		self.data.fix_alphabet()
		print('train_represents: %s' % len(self.train_represents))
		self.model = TextMatchModel(self.data)
		self.model.load_state_dict(torch.load(model_dir, map_location=self.model.configs['map_location']))
		self.model.eval()
		self.model.to(self.model.configs['device'])
		# 读取no_train文件并id化:
		self.data.no_train_texts, self.data.no_train_ids = self.data.read_no_train(no_train_path)
		# 计算no_train文件的表征:
		no_train_texts, no_train_represents, no_train_label_ids = get_represents(
			self.data, self.model, 'no_train', self.model.configs)
		print('no_train_represents: %s' % len(no_train_represents))
		print('no_train_texts: %s' % no_train_texts)
		# 汇总train_texts和no_train_texts的所有句子表征
		self.train_texts = self.data.train_texts + no_train_texts
		self.train_represents += no_train_represents
		# self.train_label_ids += no_train_label_ids

	def inference(self, text):
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
		max_score, max_similar_text = self.cal_similarity(pred_represent, self.train_represents)
		pred_text = ''.join(max_similar_text[0])
		pred_label = max_similar_text[-1]
		# 置信度、最接近的text，最接近的label
		return max_score, pred_text, pred_label

	def cal_similarity(self, pred_represent, train_represents):
		pred_represent = pred_represent.cpu().data.numpy()
		score = cosine_similarity(pred_represent, train_represents)
		max_id, max_score = np.argmax(score, axis=-1)[0], np.max(score, axis=-1)[0]
		# max_simialr_text, max_similar_label_id = self.train_texts[max_id], self.train_label_ids[max_id]
		max_simialr_text = self.train_texts[max_id]
		return max_score, max_simialr_text


if __name__ == '__main__':
	texts = ['打开房间的灯', '把空调温度调高2度', '客厅', '空调']
	texts = [
		'我回家了', 'M', '2', '110', '我不知道啊', '我查YOUYOU没出来啊', 'U', 'V', ' ', '你不用不用你唤醒了', '你能干什么', '卧室', '空调', '设备',
		'<', '>', '&', '*', '?', '什么', '咋了', '给你起个名字叫小狗子吧', '小安', '安', '安全', '查', '加', '湿', '打', '窗帘',
		'制冷', '制热', '打开蓝牙', '灯光', '是的', '声音大一点', '声音大一些', '设置温度为20', '调节温度到20', '晾衣架调高',
		'晾衣架高度调高', '客厅', '打开加热', '关闭加热功能']
	# texts = [
	# 	'打开灯', '打开电视', '把音量调高一点', '空调的湿度调低', '扫地机回去充电', '扫地机设为回充模式', '晾衣架调高一点', '客厅的灯调亮一点', '风扇调小一点',
	# 	'打开空调', '打开豆浆机', '我回来了', '回来啦', '温度调高2度', '湿度调高2度', '打开香薰机的灯', '关闭台灯的炫彩模式', '关闭空调的自动模式', '关闭静音模式',
	# 	'暂停播放', '暂停一下', '暂停电视机', '暂停房间的空调']
	# texts = ['我回家了']
	start = datetime.datetime.now()
	tm = TextMatch()
	print('model inits time costs: %s' % (datetime.datetime.now() - start).total_seconds())
	for text in texts:
		start = datetime.datetime.now()
		confidence, similar_text, pred_label = tm.inference(text)
		end = datetime.datetime.now()
		print('text:%s, pred_label:%s, confidence:%s, similar_text: %s,  time costs: %s' % (
			text, pred_label, confidence, similar_text, (end - start).total_seconds()))
