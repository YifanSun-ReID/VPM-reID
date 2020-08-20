from __future__ import absolute_import

import torch
import math
from torch import nn
import torch.nn.functional as F

class CosFaceLoss(nn.Module):
    def __init__(self, m=0, s=16, weight=None):
        super(CosFaceLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, input, target):

       
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)

        output = self.s * (input - one_hot * self.m)

        return F.cross_entropy(output, target, weight = self.weight)


