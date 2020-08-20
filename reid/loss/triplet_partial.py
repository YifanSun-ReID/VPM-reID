from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb

class PartialTripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(PartialTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    def square_dist(self, part_feat):
        n = part_feat.size(0)
        dist = torch.pow(part_feat, 2).sum(1).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, part_feat, part_feat.t())
        dist = torch.clamp(dist, min=1e-15)
        dist = torch.clamp(dist, min=1e-12).sqrt()
        return dist
     
    def forward(self, inputs, targets, part_labels, pscore):
        # For each anchor, find the hardest positive and negative
        num_parts = inputs.size(2)
        n = inputs.size(0)
        start_p = part_labels[:,2]
        end_p = part_labels[:,-3]
        flag = torch.autograd.Variable(torch.ones([n,num_parts]))
        flag = []
        for i in range(num_parts):
            flag.append(start_p.le(i)*end_p.ge(i))
        flag = torch.cat([tmp.unsqueeze(1) for tmp in flag],1)
        uflag = flag.unsqueeze(1).expand(n,n,num_parts)
        vflag = flag.unsqueeze(0).expand(n,n,num_parts)
#        pdb.set_trace()
        join_flag = uflag*vflag
        join_flag.requires_grad = False
        join_flag = join_flag.float()
        num = join_flag.sum(2)


        tt = pscore.unsqueeze(1).expand(n,n,num_parts)
        ttt = pscore.unsqueeze(0).expand(n,n,num_parts)
        wjoin =tt*ttt
        wjoin = wjoin/wjoin.sum(2,True).expand_as(wjoin)
        num.requires_grad = False
        part_dist = []
        weights = [0.6, 0.6, 0.6, 1.0, 1.5, 2.0]
        for i in range(num_parts):
#            part_dist.append(self.square_dist(inputs[:,:,i]*weights[i]).unsqueeze(2))
            part_dist.append(self.square_dist(inputs[:,:,i]).unsqueeze(2))
        part_dist = torch.cat(part_dist,2)
        part_dist = part_dist.clamp(min=1e-15)
        mask = join_flag
        mask.requires_grad = False
        dist = part_dist*(wjoin.detach())
        dist = dist.sum(2)
 

        
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
