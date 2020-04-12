# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-02-02
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(34)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = True if device.type == 'cuda' else False


class TripletLoss(nn.Module):
	def __init__(self, loss_type='batch_hard', margin=0.5):
		super(TripletLoss, self).__init__()
		self.loss_type = loss_type
		self.margin = margin

	def forward(self, labels, embeddings):
		if self.loss_type == 'batch_hard':
			loss = self.batch_hard_triplet_loss(labels, embeddings)
		else:
			loss = nn.CrossEntropyLoss()
		return loss

	def batch_hard_triplet_loss(self, labels, embeddings, squared=False):
		pairwise_distances = self._pairwise_distance(embeddings)

		mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
		mask_anchor_positive = torch.tensor(mask_anchor_positive, dtype=torch.float).to(device)
		# print('mask_anchor_positive: %s' % mask_anchor_positive)
		# print('pairwise_distances: %s' % pairwise_distances)
		anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_distances)
		hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]  # 取每一行最大的值即为最大positive距离

		mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
		mask_anchor_negative = torch.tensor(mask_anchor_negative, dtype=torch.float).to(device)
		# print('mask_anchor_negative: %s' % mask_anchor_negative)
		max_anchor_negative_dist = torch.max(pairwise_distances, dim=1, keepdim=True)[0]
		anchor_negative_dist = pairwise_distances + max_anchor_negative_dist * (torch.tensor(1.0) - mask_anchor_negative)
		hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]

		triplet_loss = torch.max(hardest_positive_dist - hardest_negative_dist + self.margin, torch.zeros(hardest_negative_dist.shape).to(device))
		# print('triplet_loss: %s' % triplet_loss)
		triplet_loss = torch.mean(triplet_loss)
		return triplet_loss

	@staticmethod
	def _get_anchor_positive_triplet_mask(labels):
		indices_equal = torch.eye(labels.shape[0]).to(device)
		# print('indices_equal: %s' % indices_equal)
		indices_not_equal = np.logical_not(indices_equal.cpu().data.numpy())  # （i, j）不相等
		labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))  # labels相等，
		mask = np.logical_and(indices_not_equal, labels_equal.cpu().data.numpy())  # 取and即可
		# print('mask: %s' % mask)
		return mask

	@staticmethod
	def _get_anchor_negative_triplet_mask(labels):
		labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
		mask = np.logical_not(labels_equal.cpu().data.numpy())
		return mask

	@staticmethod
	def _pairwise_distance(embeddings, squared=False):
		dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))
		# print('dot_product: %s' % dot_product)
		square_norm = torch.diag(dot_product)
		# print('square_norm: %s' % square_norm)
		distances = torch.unsqueeze(square_norm, dim=1) - 2.0 * dot_product + torch.unsqueeze(square_norm, dim=0)
		# print('distances:%s' % distances)
		distances = torch.max(distances, torch.zeros(distances.shape).to(device))
		# print('distances:%s' % distances)
		if not squared:
			mask = torch.tensor(torch.eq(distances, 0.0), dtype=torch.float).clone().detach().to(device)
			# print('mask: %s' % mask)
			distances = distances + mask * 1e-16
			distances = torch.sqrt(distances)
			distances = distances * (torch.tensor(1.0) - mask)
		return distances


if __name__ == '__main__':
	labels = torch.randint(low=0, high=3, size=[32])
	embeddings = torch.randn(size=[32, 256])
	triplet_loss = TripletLoss()
	# distances = triplet_loss._pairwise_distance(embeddings)
	loss = triplet_loss.batch_hard_triplet_loss(labels, embeddings)
	print(loss)
