# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-02-13
"""
import os
import pickle
import ast
import torch
import sys
import datetime
root_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
sys.path.append(root_path)
from constants import ROOT_PATH
from models.text_match_v1.utils.data import Data
from models.text_match_v1.model.ccnn_wlstm import TextMatchModel
from models.text_match_v1.utils.functions import get_represents

dset_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/alphabet.dset')
represent_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/represent.dset')
model_dir = os.path.join(ROOT_PATH, 'saved_models/text_match_v1/text_match_v1.model')

no_train_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/add/no_train.csv')
output_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data')

mode = 0
"""
0:写入pkl文件
1:写入txt文件
2:写入faiss
"""


def write_represents_to_pkl(path, output_path, name='train'):
	data = Data(ori_texts=[], labels=[], if_train=False)
	with open(path, 'rb') as rbf:
		data.char_alphabet.instance2index = pickle.load(rbf)
		data.word_alphabet.instance2index = pickle.load(rbf)
		data.label_alphabet.instance2index = pickle.load(rbf)
		data.char_alphabet_size = pickle.load(rbf)
		data.word_alphabet_size = pickle.load(rbf)
		data.label_alphabet_size = pickle.load(rbf)
		data.label_alphabet.instances = pickle.load(rbf)
		data.train_texts = pickle.load(rbf)
		data.train_ids = pickle.load(rbf)
	data.fix_alphabet()
	model = TextMatchModel(data)
	model.load_state_dict(torch.load(model_dir, map_location=model.configs['map_location']))
	model.eval()
	model.to(model.configs['device'])
	train_texts, train_represents, train_label_ids = get_represents(data, model, name, model.configs)
	# 写入
	# with open(path, 'ab') as abf:
	# 	pickle.dump(train_texts, abf)
	# 	pickle.dump(train_represents, abf)
	# 	pickle.dump(train_label_ids, abf)
	with open(output_path, 'wb') as wbf:
		pickle.dump(train_represents, wbf)


def write_represents_to_txt(path, output_path, name='train'):
	data = Data(ori_texts=[], labels=[], if_train=False)
	with open(path, 'rb') as rbf:
		data.char_alphabet.instance2index = pickle.load(rbf)
		data.word_alphabet.instance2index = pickle.load(rbf)
		data.label_alphabet.instance2index = pickle.load(rbf)
		data.char_alphabet_size = pickle.load(rbf)
		data.word_alphabet_size = pickle.load(rbf)
		data.label_alphabet_size = pickle.load(rbf)
		data.label_alphabet.instances = pickle.load(rbf)
		data.train_texts = pickle.load(rbf)
		data.train_ids = pickle.load(rbf)
	data.fix_alphabet()
	model = TextMatchModel(data)
	model.load_state_dict(torch.load(model_dir, map_location=model.configs['map_location']))
	model.eval()
	model.to(model.configs['device'])
	data.no_train_texts, data.no_train_ids = data.read_no_train(no_train_path)
	train_texts, train_represents, train_label_ids = get_represents(data, model, name, model.configs)
	if not os.path.exists(output_path + '/train_texts.txt'):
		with open(output_path + '/train_texts.txt', 'w') as wf:
			for item in train_texts:
				wf.write('%s\n' % item)
		with open(output_path + '/train_represents.txt', 'w') as wf:
			for item in train_represents:
				wf.write('%s\n' % item)
		with open(output_path + '/train_label_ids.txt', 'w') as wf:
			for item in train_label_ids:
				wf.write('%s\n' % item)


def load_represents_from_txt(path):
	train_texts, train_represents, train_label_ids = [], [], []
	with open(path + '/train_texts.txt', 'r') as rf:
		for line in rf:
			item = ast.literal_eval(line)
			train_texts.append(item)
	with open(path + '/train_represents.txt', 'r') as rf:
		for line in rf:
			item = ast.literal_eval(line)
			train_represents.append(item)
	with open(path + '/train_label_ids.txt', 'r') as rf:
		for line in rf:
			item = ast.literal_eval(line)
			train_label_ids.append(item)
	return train_texts, train_represents, train_label_ids


# def main():
# 	if mode == 0:
# 		n = 0
# 		with open(represent_path, 'rb') as rbf:
# 			while True:
# 				try:
# 					line = pickle.load(rbf)
# 					n += 1
# 				except EOFError:
# 					break
# 		if n < 10:
# 			write_represents_to_pkl(dset_path)
# 	elif mode == 1:
# 		write_represents_to_txt(dset_path, output_path)

def main():
	if mode == 0:
		write_represents_to_pkl(dset_path, represent_path)


if __name__ == '__main__':
	main()
	# s = datetime.datetime.now()
	# load_represents_from_txt(output_path)
	# print('load from txt costs: %s' % (datetime.datetime.now() - s).total_seconds())
