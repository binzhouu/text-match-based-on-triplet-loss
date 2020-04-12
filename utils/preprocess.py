# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-01-14
"""
import os
from constants import ROOT_PATH
import logging
from collections import defaultdict
import re
from pkuseg import pkuseg
from models.text_match_v1.utils.ac import AC

specific_words_path = os.path.join(ROOT_PATH, 'configs/specific_words.txt')
user_dict = os.path.join(ROOT_PATH, 'lexicon/user_dict.txt')
logger = logging.getLogger(__name__)


class PreProcess(object):
	def __init__(self):
		self.word_dict = self.read_word_onto()
		self.pkuseg = pkuseg(user_dict=user_dict)
		self.ac = AC()
		for key in self.word_dict.keys():
			self.ac.add(key)

	@classmethod
	def read_word_onto(cls):
		word_dict = {}
		word_list = ['tool', 'room', 'color_name', 'color_temperature', 'model', 'channel_name', 'show_name']
		with open(specific_words_path, 'r') as rf:
			for line in rf:
				pairs = re.split(r'[\s\t]+', line.strip())
				if pairs[0] == 'word':
					continue
				if pairs[1] in word_list:
					k, v = pairs[0], '<' + pairs[1] + '>'
					word_dict[k] = v
		# word_dict = OrderedDict(word_dict)
		return word_dict

	# 常规切词
	def pku_segment(self, text):
		"""

		:param text: '打开灯'
		:return:
		"""
		if not isinstance(text, str):
			text = str(text)
		seg_list = self.pkuseg.cut(text)  # 要保证把'<'和'>'给切出来
		return seg_list

	# 处理句子中已有<...>，但是被切开了，将'<...>'合并成一个word
	@staticmethod
	def merge_word(seg_list):
		"""

		:param seg_list: ['打开','灯']
		:return:
		"""
		merge_list = []
		flag = False
		tmp = ''
		for item in seg_list:
			if item != '<' and flag is False:
				merge_list.append(item)
			elif item == '>':
				tmp += item
				merge_list.append(tmp)
				tmp = ''
				flag = False
			elif item == '<' or flag is True:
				tmp += item
				flag = True
		return merge_list

	# 修正切词；将<...>加在业务词后面
	def generalization(self, merge_list):
		seg_bound = set()  # 每个词的首尾index，即分词的的边界
		pos = 0  # 位置滑动
		for item in merge_list:
			seg_bound.add(pos)
			seg_bound.add(pos + len(item)-1)
			pos += len(item)
		res = self.ac.search(''.join(merge_list))  # 针对text进行search,res是业务词

		new_res = defaultdict(list)  # 正确的业务词及index
		#  search出来的业务词不在分词边界的，会删除
		for word in res:  # word:业务词
			positions = res[word]  # 切词结果的边界
			positions = positions[0]
			start_idx, end_idx = positions[0], positions[-1]
			if start_idx in seg_bound and (end_idx-1) in seg_bound:
				new_res[word].append(positions)

		flag = []
		start = 0
		inc = 0
		# 基于merge_word结果的flag标记
		for seg in merge_list:
			for idx in range(start, start+len(seg)):
				flag.append(inc)  # 卧室/的/机顶盒/播放 -> [0,0,1,2,2,2,3,3]
			start += len(seg)
			inc += 1
		inc += 1
		# 将业务词的flag标记体现出来(原有基础上+1)：避免切词把业务词切错的情况
		"""
		假定：
		卧室/的/机顶盒/播放 -> [0,0,1,2,2,3,4,4]
		修正：
		卧室/的/机顶盒/播放 -> [5,5,1,6,6,6,4,4]
		"""
		for key in new_res:
			for position in new_res[key]:
				for idx in range(position[0], position[1]):
					flag[idx] = inc
				inc += 1

		ret = []
		buffer = ''
		sentence = ''.join(merge_list)

		# 根据flag标记，和merge_list结果重新切分句子，保证<...>切分出来
		for idx in range(len(sentence)):
			if idx == 0:
				buffer = sentence[idx]
			elif flag[idx] != flag[idx-1]:
				rr = self.get_map(buffer)  # 如果buffer在self.word_dict
				if type(rr) == list:
					ret.append(rr[0])  # append word
					ret.append(rr[1])  # append <...>
				else:  # buffer不在self.word_dict，就是普通词
					ret.append(rr)
				buffer = sentence[idx]
			else:
				buffer += sentence[idx]
			if idx == len(sentence) - 1:
				rr = self.get_map(buffer)
				if type(rr) == list:
					ret.append(rr[0])
					ret.append(rr[1])
				else:
					ret.append(rr)
		return ret

	def get_map(self, buffer):
		if buffer in self.word_dict:
			return [buffer, self.word_dict[buffer]]
		return buffer


if __name__ == '__main__':
	pre_process = PreProcess()
	text = '客厅空气净化器的PM2.5是多少'
	# seg_list = pre_process.pku_segment('把台灯<tool>调成暖白<color_temperature>好吗？')
	# seg_list = pre_process.pku_segment('打开主卧的灯吧')
	seg_list = pre_process.pku_segment(text)
	merge_list = pre_process.merge_word(seg_list)
	print(merge_list)
	pre_process.generalization(merge_list)
