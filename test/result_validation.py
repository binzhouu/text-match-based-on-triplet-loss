# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-02-28
"""

import os
import re

# backup_file = os.path.join(os.path.dirname(__file__), 'backup.txt')
faiss_result_file = os.path.join(os.path.dirname(__file__), 'faiss_result.txt')
faiss_result2_file = os.path.join(os.path.dirname(__file__), 'faiss_result2.txt')


backup_list, faiss_result_list, faiss_result2_list = [], [], []
pattern = re.compile(r'(?P<time>time costs:.+)')

# with open(backup_file, 'r') as rf:
# 	for line in rf:
# 		line = pattern.sub('', line)
# 		backup_list.append(line.strip())

with open(faiss_result_file, 'r') as rf:
	for line in rf:
		line = pattern.sub('', line)
		faiss_result_list.append(line.strip())

with open(faiss_result2_file, 'r') as rf:
	for line in rf:
		line = pattern.sub('', line)
		faiss_result2_list.append(line.strip())

for m, n in zip(faiss_result_list, faiss_result2_list):
	if m != n:
		print(m)
		print(n)
