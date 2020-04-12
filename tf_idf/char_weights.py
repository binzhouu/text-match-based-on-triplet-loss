# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-06
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import joblib
from constants import ROOT_PATH
from models.text_match_v1.utils.data import Data

dset_path = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/data/alphabet.dset')
model_path = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/char_weight/char_weight.joblib')


class CharWeight(object):
	def __init__(self, if_train=False):
		if if_train:
			self.data = Data(ori_texts=[], labels=[], if_train=False)
			with open(dset_path, 'rb') as rbf:
				self.data.char_alphabet.instance2index = pickle.load(rbf)
				_ = pickle.load(rbf)
				_ = pickle.load(rbf)
				_ = pickle.load(rbf)
				_ = pickle.load(rbf)
				_ = pickle.load(rbf)
				_ = pickle.load(rbf)
				self.data.train_texts = pickle.load(rbf)
			# vocabulary中增加一个unk,作为OOV的处理
			if self.data.char_alphabet.get_instance(0) is None:
				self.data.char_alphabet.instance2index['unk'] = 0

	def train(self):
		train_texts = []
		for train_text in self.data.train_texts:
			intent = train_text[-1]
			# if intent:
			if intent in ['turn_on', 'turn_off']:
				text = ''
				words = train_text[0]
				for word in words:
					if not self.data.specific_word(word):
						text += word
				text = ' '.join(text)
				train_texts.append(text)

		vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', vocabulary=self.data.char_alphabet.instance2index)
		vectorizer.fit(train_texts)
		# 实例化权重模型
		joblib.dump(vectorizer, model_path)

	@classmethod
	def load_model(cls, model_dir):
		vectorizer = joblib.load(model_dir)
		return vectorizer


if __name__ == '__main__':
	char_weight = CharWeight(if_train=True)
	char_weight.train()
	# vecm = char_weight.load_model(model_path)
	# res = vecm.transform(['打 开 回 家 场 景 unk'])

	# vectorizer = CharWeight.load_model(model_path)
	# res = vectorizer.transform(['拨 打 0 0 0 0 0 0 0 0 0'])
	# res = vectorizer.transform(['打 开 unk 场 景'])
	# print(res)
