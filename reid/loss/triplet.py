from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(1).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#        dist = dist.clamp(min=1e-12)  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
#            dist_an.append(dist[i][1 - mask[i]].min())
            dist_an.append(dist[i][Variable(torch.ones(mask.size(0)).byte().cuda()) - mask[i]].min().unsqueeze(0))
#            dist_an.append(dist[i][Variable(torch.ones(mask.size(0)).byte().cuda()) - mask[i]].max())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
#        dist_ap = dist_ap.max()
#        dist_an = dist_an.min()
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
