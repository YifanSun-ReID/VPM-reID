from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # pdb.set_trace()
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        labels = np.random.randint(0, 2, n)
        eye = Variable(1 - torch.eye(n).byte()).cuda()
        mask_ap = mask * eye
        # pdb.set_trace()
        for i in range(n):
            if labels[i] == 1:
                temp = dist[i][mask_ap[i]]
                idx = np.random.randint(0, temp.size(0))
                dist_ap.append(temp[idx])
            else:
                temp = dist[i][mask[i] == 0]
                idx = np.random.randint(0, temp.size(0))
                dist_an.append(temp[idx])
        # pdb.set_trace()
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # compute loss
        loss = (1.0 * dist_ap.sum() + 1.0 * torch.clamp(self.margin - dist_an, min=0.0).sum()) / n
        prec = ((dist_an.data > self.margin).sum() + (dist_ap.data < self.margin).sum()) * 1./n
        # Compute ranking hinge loss
        #y = dist_an.data.new()
        #y.resize_as_(dist_an.data)
        #y.fill_(1)
        #y = Variable(y)
        #loss = self.ranking_loss(dist_an, dist_ap, y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
