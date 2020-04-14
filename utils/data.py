# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-01-13
"""
import os
import sys
import re
import logging
import pandas as pd
import pickle
import yaml
import random
root_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-3])
# print(root_path)
sys.path.append(root_path)
from constants import ROOT_PATH
from models.text_match_v1.utils.preprocess import PreProcess
from models.text_match_v1.utils.alphabet import Alphabet
from models.text_match_v1.utils.functions import normalize_word

random.seed(34)

# data_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/notebook/data_v1.csv')
data_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/notebook/data_v2.csv')
match_yml_path = os.path.join(ROOT_PATH, 'models/text_match_v1/conf/match.yml')

with open(match_yml_path, 'r') as rf:
	match_configs = yaml.load(rf, Loader=yaml.FullLoader)

logger = logging.getLogger(__name__)


class Data(object):
	def __init__(self, ori_texts, labels, if_train=True):
		self.match_configs = match_configs
		self.ori_texts, self.labels = ori_texts, labels
		self.pre_process = PreProcess()  # 顺带初始化字典树
		self.seg_lists = self.segment(self.ori_texts)
		self.char_alphabet = Alphabet('char')
		self.word_alphabet = Alphabet('word')
		self.label_alphabet = Alphabet('label', label=True)
		self.char_alphabet_size, self.word_alphabet_size, self.label_alphabet_size = -1, -1, -1
		if if_train:
			logger.info('Data build_alphabet start..')
			self.char_lists = self.build_alphabet()
			logger.info('Data read_instance start..')
			self.texts, self.ids = self.read_instance(self.seg_lists, self.labels)
			logger.info('Data sample_split start..')
			self.train_texts, self.train_ids, self.dev_texts, self.dev_ids, self.test_texts, self.test_ids, \
				self.train_label_ids = self.sample_split()

	@classmethod
	def read_data(cls):
		data_df = pd.read_csv(data_path)
		ori_texts = list(data_df.loc[:, 'text'])
		labels = list(data_df.loc[:, 'target'])
		return cls(ori_texts, labels)

	def read_no_train(self, no_train_path, mode='add'):
		no_train_df = pd.read_csv(no_train_path)
		no_train_texts = list(no_train_df.loc[:, 'text'])
		no_train_labels = list(no_train_df.loc[:, 'target'])
		no_train_seg_lists = self.segment(no_train_texts)
		texts, ids = self.read_instance(no_train_seg_lists, no_train_labels, mode)
		return texts, ids

	def read_scene_text_list(self, text_list):
		chars, ids = [], []
		for sentence in text_list:
			sen_text = [char for char in sentence]
			sen_id = [self.char_alphabet.get_index(normalize_word(char)) for char in sentence]
			chars.append(sen_text)
			ids.append(sen_id)
		return chars, ids

	def segment(self, ori_texts):
		seg_lists = []
		for ori_text in ori_texts:
			seg_list = self.pre_process.pku_segment(ori_text)
			merge_list = self.pre_process.merge_word(seg_list)
			word_list = self.pre_process.generalization(merge_list)
			seg_lists.append(word_list)
		return seg_lists

	def build_alphabet(self):
		char_lists = []
		for seg_list, label in zip(self.seg_lists, self.labels):
			char_list = []
			for word in seg_list:
				# word = normalize_word(word)
				self.word_alphabet.add(normalize_word(word))
				if self.specific_word(word):
					self.char_alphabet.add(word)
					char_list.append(word)
				else:
					for char in word:
						char_list.append(char)
						char = normalize_word(char)
						self.char_alphabet.add(char)
			char_lists.append(char_list)
			self.label_alphabet.add(label)
		self.char_alphabet_size = self.char_alphabet.size()
		self.word_alphabet_size = self.word_alphabet.size()
		self.label_alphabet_size = self.label_alphabet.size()
		self.fix_alphabet()
		return char_lists

	def read_instance(self, seg_lists, labels, mode='train'):
		texts, ids = [], []
		for seg_list, label in zip(seg_lists, labels):
			char_list, char_id_list, word_id_list, = [], [], []
			label_id = self.label_alphabet.get_index(label, mode)
			for word in seg_list:
				word_id = self.word_alphabet.get_index(normalize_word(word))
				word_id_list.append(word_id)
				chars, char_ids = [], []
				if self.specific_word(word):
					chars.append(word)
					char_ids.append(self.char_alphabet.get_index(normalize_word(word)))
				else:
					for char in word:
						chars.append(char)
						char_ids.append(self.char_alphabet.get_index(normalize_word(char)))
				char_list.append(chars)
				char_id_list.append(char_ids)
			# for char in char_list:
			# 	char_id = self.char_alphabet.get_index(normalize_word(char))
			# 	char_id_list.append(char_id)
			texts.append([seg_list, char_list, label])
			ids.append([word_id_list, char_id_list, label_id])
		return texts, ids

	# 如果word是<...>则返回True -- 看成一个char
	@staticmethod
	def specific_word(word):
		if word[0] == '<' and word[-1] == '>':
			return True
		else:
			return False

	def fix_alphabet(self):
		self.char_alphabet.close()
		self.word_alphabet.close()
		self.label_alphabet.close()

	def sample_split(self):
		batch_size = match_configs['batch_size']
		sampling_rate = match_configs['sampling_rate']
		train_texts, train_ids, dev_texts, dev_ids, test_texts, test_ids = [], [], [], [], [], []
		train_label_ids = []
		for label_name in self.label_alphabet.instances:
			indexes = [n for n in range(len(self.labels)) if self.labels[n] == label_name]
			random.shuffle(indexes)
			one_class_texts = [self.texts[i] for i in indexes]
			one_class_ids = [self.ids[i] for i in indexes]
			n = int(len(one_class_ids) * sampling_rate)  # 每一个类别中的抽样比例(取整数)
			one_class_dev_texts, one_class_dev_ids = one_class_texts[:n], one_class_ids[:n]
			one_class_test_texts, one_class_test_ids = one_class_texts[n:2*n], one_class_ids[n:2*n]
			one_class_train_texts, one_class_train_ids = one_class_texts[2*n:],  one_class_ids[2*n:]
			one_class_train_label_ids = [self.label_alphabet.get_index(label_name)] * len(one_class_train_texts)  # 训练前抽样会用到
			train_texts += one_class_train_texts
			train_ids += one_class_train_ids
			dev_texts += one_class_dev_texts
			dev_ids += one_class_dev_ids
			test_texts += one_class_test_texts
			test_ids += one_class_test_ids
			train_label_ids += one_class_train_label_ids

		alphabet_path = os.path.join(ROOT_PATH, self.match_configs['alphabet_path'])
		# print(train_texts[0])
		# print(train_texts[1])
		if not os.path.exists(os.path.join(alphabet_path)):
			with open(alphabet_path, 'wb') as wbf:
				pickle.dump(self.char_alphabet.instance2index, wbf)
				pickle.dump(self.word_alphabet.instance2index, wbf)
				pickle.dump(self.label_alphabet.instance2index, wbf)
				pickle.dump(self.char_alphabet_size, wbf)
				pickle.dump(self.word_alphabet_size, wbf)
				pickle.dump(self.label_alphabet_size, wbf)
				pickle.dump(self.label_alphabet.instances, wbf)
				pickle.dump(train_texts, wbf)
				pickle.dump(train_ids, wbf)  # 将train_ids也保存到alphabet到pickle文件中：
		return train_texts, train_ids, dev_texts, dev_ids, test_texts, test_ids, train_label_ids


if __name__ == '__main__':
	data = Data.read_data()
