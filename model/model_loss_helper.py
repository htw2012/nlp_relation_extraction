#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title= ""
author= "huangtw"
mtime= 2018/3/17
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from keras import backend as K
def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)

def one_hot(indices, depth, on_value=1, off_value=0):
    np_ids = np.array(indices.cpu().data.numpy()).astype(int)
    if len(np_ids.shape) == 2:
        encoding = np.zeros([np_ids.shape[0], np_ids.shape[1], depth], dtype=int)
        added = encoding + off_value
        for i in range(np_ids.shape[0]):
            for j in range(np_ids.shape[1]):
                added[i, j, np_ids[i, j]] = on_value
        return Variable(torch.FloatTensor(added.astype(float))).cuda()
    if len(np_ids.shape) == 1:
        encoding = np.zeros([np_ids.shape[0], depth], dtype=int)
        added = encoding + off_value
        for i in range(np_ids.shape[0]):
            added[i, np_ids[i]] = on_value
        return Variable(torch.FloatTensor(added.astype(float))).cuda()

class NovelDistanceLoss(nn.Module):
    def __init__(self, nr, margin=1):
        super(NovelDistanceLoss, self).__init__()
        self.nr = nr
        self.margin = margin

    def forward(self, wo, rel_weight, in_y):
        wo_norm = F.normalize(wo)  # (bz, dc)
        bz = wo_norm.data.size()[0]
        dc = wo_norm.data.size()[1]
        wo_norm_tile = wo_norm.view(-1, 1, dc).repeat(1, self.nr, 1)  # (bz, nr, dc)
        batched_rel_w = F.normalize(rel_weight).view(1, self.nr, dc).repeat(bz, 1, 1)
        all_distance = torch.norm(wo_norm_tile - batched_rel_w, 2, 2)  # (bz, nr, 1)
        mask = one_hot(in_y, self.nr, 1000, 0)  # (bz, nr)
        masked_y = torch.add(all_distance.view(bz, self.nr), mask)
        neg_y = torch.min(masked_y, dim=1)[1]  # (bz,)
        neg_y = torch.mm(one_hot(neg_y, self.nr), rel_weight)  # (bz, nr)*(nr, dc) => (bz, dc)

        pos_y = torch.mm(one_hot(in_y, self.nr), rel_weight)

        neg_distance = torch.norm(wo_norm - F.normalize(neg_y), 2, 1)
        pos_distance = torch.norm(wo_norm - F.normalize(pos_y), 2, 1)
        loss = torch.mean(pos_distance + self.margin - neg_distance)

        return loss