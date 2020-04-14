# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-01-13
"""


class Alphabet(object):
    def __init__(self, name, label=False, keep_growing=True):
        self.name = name
        self.UNKNOWN = '</unk>'
        self.label = label
        self.instance2index = dict()
        self.instances = list()
        self.keep_growing = keep_growing

        self.default_index = 0
        self.next_index = 1
        # 将'</unk>'放在alphabet放在第一个
        if not self.label:
            self.add(self.UNKNOWN)

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def size(self):
        return len(self.instances) + 1

    def get_index(self, instance, mode='train'):
        # 当出现新当label时当处理方式:
        if mode == 'add' and self.label and instance not in self.instance2index:
            self.keep_growing = True
            self.next_index = len(self.instances) + 1
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        if index == 0:
            if self.label:
                return self.instances[0]
            # First index is occupied by the wildcard element.
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance, return the first label.')
            return self.instances[0]

    def iteritems(self):
        return self.instance2index.items()

    def close(self):
        self.keep_growing = False
