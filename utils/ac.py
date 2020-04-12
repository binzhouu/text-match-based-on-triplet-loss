# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-01-13
"""
from collections import defaultdict


class Node(object):
	def __init__(self):
		self.next = {}
		self.fail = None
		self.isWord = False
		self.word = ''


class AC(object):
	def __init__(self):
		self.root = Node()

	def add(self, word):
		temp_root = self.root
		for char in word:
			if char not in temp_root.next:
				temp_root.next[char] = Node()
			temp_root = temp_root.next[char]
		temp_root.isWord = True
		temp_root.word = word

	def search(self, content):  # content是个句子
		p = self.root
		result = defaultdict(list)
		currentposition = 0

		while currentposition < len(content):
			word = content[currentposition]
			# 用不到，删除
			# while word in p.next == False and p != self.root:
			# 	p = p.fail
			if word in p.next:
				p = p.next[word]  # 这里的p仍然是一个node()
			else:
				p = self.root
				if word in p.next:
					p = p.next[word]
			if p.isWord:
				# 把<...>到start和end加入
				result[p.word].append((currentposition - len(p.word) + 1, currentposition + 1))
			currentposition += 1
		return dict(result)


if __name__ == '__main__':
	ac = AC()
	ac.add('房间')
	ac.add('电视')
	ac.add('电视机')
	ac.add('电风扇')
	ac.add('晾衣架')
	# ac.search('打开电视把')
	# ac.search('打开房间电视啊')
	ac.search('电视机')
