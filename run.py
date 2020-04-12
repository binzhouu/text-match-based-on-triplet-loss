# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-01-13
"""
import logging.config
import os
import sys
import torch.optim as optim
import datetime
import time
from collections import Counter
import numpy as np
import pandas as pd
import random
import torch
import ast
from tensorboardX import SummaryWriter
root_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
sys.path.append(root_path)
from constants import ROOT_PATH
from models.text_match_v1.utils.data import Data
from models.text_match_v1.model.ccnn_wlstm import TextMatchModel
from models.text_match_v1.utils.functions import batchfy_classification_with_label, get_represents, evalute
from models.text_match_v1.save_train_represents import main

logger = logging.getLogger(__name__)
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

writer = SummaryWriter('./tensorboard_log')

data_source_path = os.path.join(ROOT_PATH, 'models/data_source/csv/zhh_0327.csv')
data_match_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/notebook/text_match_v1.csv')
output_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/notebook/data_v1.csv')

data_source_path = os.path.join(ROOT_PATH, 'models/data_source/Notebook/niudi.txt')
output_path = os.path.join(ROOT_PATH, 'models/text_match_v1/data/notebook/data_v2.csv')


class Run(object):
	def __init__(self):
		# 生成匹配模型的训练语料
		# self.generate_corpus()
		self.generate_csv(data_source_path, output_path)

	def train(self):
		data = Data.read_data()
		match_configs = data.match_configs
		model = TextMatchModel(data)
		model_configs = model.configs
		if model_configs['gpu']:
			model = model.cuda()
		batch_size = match_configs['batch_size']
		model_configs.update({'batch_size': batch_size})
		optimizer = optim.Adam(model.parameters(), lr=model_configs['lr'], weight_decay=model_configs['l2'])
		if model_configs['gpu']:
			model = model.cuda()
		best_dev = -10
		last_improved = 0
		logger.info('train start:%s', datetime.datetime.now())

		for idx in range(model_configs['epoch']):
			epoch_start = time.time()
			temp_start = epoch_start
			logger.info('Epoch: %s/%s' % (idx, model_configs['epoch']))
			optimizer = self.lr_decay(optimizer, idx, model_configs['lr_decay'], model_configs['lr'])

			sample_loss = 0
			total_loss = 0
			# right_token = 0
			# whole_token = 0
			logging.info("first input word _list: %s, %s" % (data.train_texts[0][1], data.train_ids[0][1]))

			model.train()
			model.zero_grad()
			num_classes_per_batch = match_configs['num_classes_per_batch']  # 16
			num_sentences_per_class = batch_size // num_classes_per_batch  # 4
			assert type(num_sentences_per_class) == int
			start = datetime.datetime.now()
			instances, total_batch = self.get_instances(data, num_classes_per_batch, num_sentences_per_class)
			logger.info('get_instances costs: %s' % (datetime.datetime.now() - start).total_seconds())
			logger.info('total_batch: %s' % total_batch)

			train_num = len(instances)
			for batch_id in range(total_batch):
				start = batch_id * batch_size
				end = (batch_id + 1) * batch_size  # end最后一定等于train_num
				instance = instances[start:end]
				# print(start, end)
				if not instance:
					continue
				# instance -> (word, char, label)
				batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, mask, batch_label = \
					batchfy_classification_with_label(instance, model_configs['gpu'], if_train=True)
				loss, sen_represent = model(
					batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, batch_label)

				sample_loss += loss.item()
				total_loss += loss.item()
				# 每10个batch，输出一下结果
				if end % (batch_size * 10) == 0:
					temp_time = time.time()
					temp_cost = temp_time - temp_start
					temp_start = temp_time
					logger.info("Instance: %s; Time: %.2fs; loss: %.4f" % (end, temp_cost, sample_loss))
					if sample_loss > 1e4 or str(sample_loss) == 'nan':
						raise ValueError("ERROR: LOSS EXPLOSION (>1e4) !")
					sample_loss = 0

				loss.backward()
				optimizer.step()
				model.zero_grad()
			temp_time = time.time()
			temp_cost = temp_time - temp_start
			logger.info("Instance: %s; Time: %.2fs; loss: %.4f" % (end, temp_cost, sample_loss))
			epoch_finish = time.time()
			epoch_cost = epoch_finish - epoch_start
			logger.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
				idx, epoch_cost, train_num / epoch_cost, total_loss))
			if total_loss > 1e4 or str(total_loss) == 'nan':
				raise ValueError("ERROR: LOSS EXPLOSION (>1e4) !")

			writer.add_scalar('Train_loss', total_loss, idx)

			# 计算当前模型下，训练集的句子表征
			_, train_represents, train_labels = get_represents(data, model, 'train', model_configs)
			# dev的验证
			speed, acc, p, r, f, _, _ = evalute(data, model, 'dev', model_configs, train_represents, train_labels)
			dev_finish = time.time()
			dev_cost = dev_finish - epoch_finish
			current_score = acc
			writer.add_scalar('Dev_acc', current_score, idx)
			writer.add_scalar('Dev_f1', f, idx)
			logger.info(
				"Dev: time: %.2fs speed: %.2fst/s; acc: %.4f weighted_avg_f1: %.4f" % (dev_cost, speed, acc, f))
			if current_score > best_dev:
				logger.info("Exceed previous best acc score: %s" % best_dev)
				model_name = os.path.join(ROOT_PATH, model_configs['model_path'] + '.model')
				torch.save(model.state_dict(), model_name)
				best_dev = current_score
				last_improved = idx

			# test的验证
			speed, acc, p, r, f, _, _ = evalute(data, model, 'test', model_configs, train_represents, train_labels)
			test_finish = time.time()
			test_cost = test_finish - dev_finish
			writer.add_scalar('Test_acc', acc, idx)
			writer.add_scalar('Test_f1', f, idx)
			logger.info(
				"Test: time: %.2fs, speed: %.2fst/s; acc: %.4f weighted_avg_f1: %.4f" % (test_cost, speed, acc, f))
			# early_stopping
			if idx - last_improved > model_configs['require_improvement']:
				logger.info('No optimization for %s epoch, auto-stopping' % model_configs['require_improvement'])
				writer.close()

				# 将所有训练样本的表征写入represent.dset
				main()
				break
		writer.close()

	@staticmethod
	def lr_decay(optimzer, epoch, decay_rate, init_lr):
		lr = init_lr / (1 + decay_rate * epoch)
		logging.info("Learning rate is set as: %s", lr)
		for param_group in optimzer.param_groups:
			param_group['lr'] = lr
		return optimzer

	# total_batch = 样本最多的类别数量 / 4 +1
	@staticmethod
	def get_instances(data, num_classes_per_batch, num_sentences_per_class):

		def get_one_class_samples(label_id):
			s = data.train_label_ids.index(label_id)
			for index in range(s, len(data.train_label_ids)):
				e = index
				if data.train_label_ids[index] != label_id or index == len(data.train_label_ids)-1:
					return data.train_ids[s:e]

		max_class_count = sorted(Counter(data.train_label_ids).items(), key=lambda x: x[1], reverse=True)[0][-1]
		num_class = data.label_alphabet_size - 1
		logger.info('max_class_count: %s, num_class: %s' % (max_class_count, num_class))
		# 迭代完所有的类别所需要的次数
		if num_class % num_classes_per_batch == 0:
			epoch_class_list = list(set(data.train_label_ids))
		else:
			epoch_class_list = list(set(data.train_label_ids))
			size = (num_class // 16 + 1) * 16 - num_class
			np.random.choice(epoch_class_list, size, replace=False)
			epoch_class_list += np.random.choice(epoch_class_list, size, replace=False).tolist()
		random.shuffle(epoch_class_list)  # 每一轮对epoch_class_list都不一样
		logger.info('epoch_class_list: %s, len: %s' % (epoch_class_list, len(epoch_class_list)))
		# 迭代完最多类别的样本需要的次数
		if max_class_count % num_sentences_per_class == 0:
			num_max_class_batch = max_class_count // num_sentences_per_class
		else:
			num_max_class_batch = max_class_count // num_sentences_per_class + 1
		logger.info('num_max_class_batch: %s' % num_max_class_batch)

		# 控制num_max_class_batch的数量，避免total_batch的值过大
		if num_max_class_batch > 3000:
			num_max_class_batch = num_max_class_batch // 10 + 1
		logger.info('After: num_max_class_batch: %s' % num_max_class_batch)

		batch_samples = []
		total_batch = 0
		for idx in range(num_max_class_batch):  # 迭代完68个样本的次数，即68//4=17
			start, end = 0, num_classes_per_batch
			for iter_idx in range(len(epoch_class_list) // num_classes_per_batch):  # 迭代完所有的class，即128//16=8
				one_batch_class_list = epoch_class_list[start:end]
				one_batch_samples = []
				for label_id in one_batch_class_list:
					# 取得一个类别的所有样本
					one_class_samples = get_one_class_samples(label_id)
					# print(label_id, one_class_samples)
					# 抽取逻辑：对一个类别对所有样本，随机抽取4个
					random_index = np.random.choice(len(one_class_samples), num_sentences_per_class, replace=False).tolist()
					one_class_random_samples = [one_class_samples[i] for i in random_index]
					one_batch_samples += one_class_random_samples  # 每次4个样本
				# 每次获得64个样本
				total_batch += 1
				batch_samples += one_batch_samples
				start = end
				end += num_classes_per_batch
			# 	print('iter_idx: %s' % iter_idx)
			# print('idx: %s' % idx)
		return batch_samples, total_batch

	@staticmethod
	def generate_corpus():
		texts, intents = [], []

		zhh_df = pd.read_csv(data_source_path)
		# for index, row in zhh_df.iterrows():
		# 	text = row['texts']
		# 	intent = row['intents']
		# 	if text not in texts:
		# 		texts.append(text)
		# 		intents.append(intent)
		texts, intents = list(zhh_df.loc[:, 'texts']), list(zhh_df.loc[:, 'intents'])

		match_df = pd.read_csv(data_match_path)
		for index, row in match_df.iterrows():
			text = row['text']
			intent = row['target']
			if text not in texts:
				texts.append(text)
				intents.append(intent)

		logger.info('Intents lass Counter: ' + '\n')
		for pairs in sorted(Counter(intents).items(), key=lambda x: x[1], reverse=True):
			logger.info(pairs)
		logger.info('All intents count: %s' % (len(set(intents))))
		data_df = pd.DataFrame({'text': texts, 'target': intents})
		data_df.to_csv(output_path, index=False)

	@staticmethod
	def generate_csv(input_file, output_file):
		texts, intents = [], []
		with open(input_file, 'r') as rf:
			for line in rf:
				line = ast.literal_eval(line)
				text = line['text']
				intent = line['intent']
				texts.append(text)
				intents.append(intent)
		for pairs in sorted(Counter(intents).items(), key=lambda x: x[1], reverse=True):
			logger.info(pairs)
		data_df = pd.DataFrame({'text': texts, 'target': intents})
		data_df.to_csv(output_file, index=False)
		logger.info('generate data.csv in: %s' % output_file)


if __name__ == '__main__':
	run = Run()
	run.train()
