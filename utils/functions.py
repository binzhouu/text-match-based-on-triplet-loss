# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-01-14
"""
import torch
import time
import numpy as np
import logging
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from constants import ROOT_PATH

logger = logging.getLogger(__name__)
glove_emb_path = os.path.join(ROOT_PATH, 'models/glove_emb/glove_ch_vec.pkl')


def normalize_word(word):
	new_word = ""
	for char in word:
		if char.isdigit():  # 如果字符串为数字组成，则为True
			# print('char:', char)
			new_word += '0'
		# print('new_word:', new_word)
		else:
			new_word += char
	return new_word


def get_represents(data, model, name, config):
	if name == 'train':
		instance_texts, instance_ids = data.train_texts, data.train_ids
	elif name == 'dev':
		instance_texts, instance_ids = data.dev_texts, data.dev_ids
	elif name == 'test':
		instance_texts, instance_ids = data.test_texts, data.test_ids
	elif name == 'add':
		instance_texts, instance_ids = data.no_train_texts, data.no_train_ids
	elif name == 'scene':
		instance_texts, instance_ids = data.scene_texts, data.scene_ids
	else:
		raise ValueError("Data name %s is not exists !" % name)

	model.eval()
	texts, represents, label_ids = [], [], []
	n = 0
	train_num = len(instance_ids)
	for instance_text, instance_id in zip(instance_texts, instance_ids):
		batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, mask, batch_label = \
			batchfy_classification_with_label([instance_id], config['gpu'], if_train=False)
		sen_represent = model(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
		sen_represent = sen_represent.cpu().data.numpy().tolist()
		batch_label = batch_label.cpu().data.numpy().tolist()
		represents += sen_represent
		label_ids += batch_label
		texts.append(instance_text)
		n += 1
		if n % 50000 == 0:
			logger.info('calculate %s / %s' % (n, train_num))
	return texts, represents, label_ids


def evalute(data, model, name, config, train_represents, train_labels):
	start_time = time.time()
	logger.info('Evaluting start, data name: %s' % name)
	_, pred_represents, gold_labels = get_represents(data, model, name, config)
	logger.info('pred_represents: %s, train_represents: %s' % (np.shape(pred_represents), np.shape(train_represents)))
	scores = cosine_similarity(pred_represents, train_represents)
	max_ids, max_scores = np.argmax(scores, axis=-1), np.max(scores, axis=-1)
	pred_labels = []
	for max_id in max_ids:
		# 预测出的最相近的句子，及最相近的句子对应的标签
		max_similar_represent, max_similar_label = train_represents[max_id], train_labels[max_id]
		pred_labels.append(max_similar_label)
	correct_num = np.sum(np.equal(pred_labels, gold_labels)).tolist()
	total_num = len(gold_labels)
	acc = correct_num/total_num

	decode_time = time.time() - start_time
	speed = total_num/decode_time
	p, r, f, _ = metrics.classification_report(pred_labels, gold_labels, output_dict=True)['weighted avg'].values()
	return speed, acc, p, r, f, pred_labels, max_scores

# def recover_sen_order(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
# 	pred_represent = pred_variable[word_recover]
# 	gold_tag = gold_variable[word_recover]
# 	mask_variable = mask_variable[word_recover]
#
# 	pred_represent = pred_represent.cpu().data.numpy().tolist()
# 	gold_tag = gold_tag.cpu().data.numpy().tolist()
# 	return pred_represent, gold_tag


def batchfy_classification_with_label(input_batch_list, gpu, if_train=True):
	batch_size = len(input_batch_list)
	words = [sent[0] for sent in input_batch_list]
	chars = [sent[1] for sent in input_batch_list]
	labels = [sent[-1] for sent in input_batch_list]
	word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)
	max_seq_len = word_seq_lengths.max().item()
	word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
	# label_seq_tensor = torch.zeros((batch_size,), requires_grad=if_train).long()

	mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
	label_seq_tensor = torch.tensor(labels, dtype=torch.long)

	for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
		seqlen = seqlen.item()
		word_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
		mask[idx, :seqlen] = torch.tensor([1] * seqlen)
	word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
	word_seq_tensor = word_seq_tensor[word_perm_idx]
	label_seq_tensor = label_seq_tensor[word_perm_idx]
	mask = mask[word_perm_idx]

	pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
	length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
	max_word_len = max(map(max, length_list))
	char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
	char_seq_lengths = torch.tensor(length_list, dtype=torch.long)
	for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
		for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
			char_seq_tensor[idx, idy, :wordlen] = torch.tensor(word)

	char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
	char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
	char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
	char_seq_tensor = char_seq_tensor[char_perm_idx]
	_, char_seq_recover = char_perm_idx.sort(0, descending=False)
	_, word_seq_recover = word_perm_idx.sort(0, descending=False)
	if gpu:
		word_seq_tensor = word_seq_tensor.cuda()
		word_seq_lengths = word_seq_lengths.cuda()
		word_seq_recover = word_seq_recover.cuda()
		label_seq_tensor = label_seq_tensor.cuda()
		char_seq_tensor = char_seq_tensor.cuda()
		char_seq_recover = char_seq_recover.cuda()
		mask = mask.cuda()
	return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, mask, \
		label_seq_tensor


def predict_batchfy_classification_with_label(input_batch_list, gpu, if_train=True):
	batch_size = len(input_batch_list)
	words = [sent[0] for sent in input_batch_list]
	chars = [sent[1] for sent in input_batch_list]
	# labels = [sent[-1] for sent in input_batch_list]
	word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)
	max_seq_len = word_seq_lengths.max().item()
	word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
	# label_seq_tensor = torch.zeros((batch_size,), requires_grad=if_train).long()

	mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
	# label_seq_tensor = torch.tensor(labels, dtype=torch.long)

	for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
		seqlen = seqlen.item()
		word_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
		mask[idx, :seqlen] = torch.tensor([1] * seqlen)
	word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
	word_seq_tensor = word_seq_tensor[word_perm_idx]
	# label_seq_tensor = label_seq_tensor[word_perm_idx]
	mask = mask[word_perm_idx]

	pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
	length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
	max_word_len = max(map(max, length_list))
	char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
	char_seq_lengths = torch.tensor(length_list, dtype=torch.long)
	for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
		for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
			char_seq_tensor[idx, idy, :wordlen] = torch.tensor(word)

	char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
	char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
	char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
	char_seq_tensor = char_seq_tensor[char_perm_idx]
	_, char_seq_recover = char_perm_idx.sort(0, descending=False)
	_, word_seq_recover = word_perm_idx.sort(0, descending=False)
	if gpu:
		word_seq_tensor = word_seq_tensor.cuda()
		word_seq_lengths = word_seq_lengths.cuda()
		word_seq_recover = word_seq_recover.cuda()
		# label_seq_tensor = label_seq_tensor.cuda()
		char_seq_tensor = char_seq_tensor.cuda()
		char_seq_recover = char_seq_recover.cuda()
		mask = mask.cuda()
	return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, \
		mask


def build_pretrain_embedding(embedding_path, word_alphabet, norm=True):
	# embedd_dict = dict()
	embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
	alphabet_size = len(word_alphabet.instance2index) + 1
	scale = np.sqrt(3.0/embedd_dim)
	pretrain_emb = np.nan_to_num(np.empty([alphabet_size, embedd_dim]))
	perfect_match = 0
	case_match = 0
	not_match = 0
	for word, index in word_alphabet.iteritems():
		if word in embedd_dict:
			if norm:
				pretrain_emb[index, :] = norm2one(embedd_dict[word])
			else:
				pretrain_emb[index, :] = embedd_dict[word]
			perfect_match += 1
		elif word.lower() in embedd_dict:
			if norm:
				pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
			else:
				pretrain_emb[index, :] = embedd_dict[word.lower()]
			case_match += 1
		else:
			pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
			not_match += 1
	pretrained_size = len(embedd_dict)
	print("Embedding:\n     pretrain word:%s, perfect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
		pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
	return pretrain_emb, embedd_dim


def load_pretrain_emb(embedding_path):
	embedd_dim = -1
	with open(embedding_path, 'rb') as rbf:
		embedd_dict = pickle.load(rbf)
		embedd_dim = embedd_dict[list(embedd_dict.keys())[0]].size
	return embedd_dict, embedd_dim


def norm2one(vec):
	root_sum_square = np.sqrt(np.sum(np.square(vec)))
	return vec / root_sum_square


if __name__ == '__main__':
	# res = normalize_word('25')
	pass
