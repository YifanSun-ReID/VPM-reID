# -*- coding: UTF-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
import pdb



def generate_column_labels(targets, ratio, start_ratio, num_parts):
    targets = torch.cat([targets.unsqueeze(1)]*(num_parts+1),1)
#    targets[:,0:6] = 751
    optargets = Variable(torch.FloatTensor(range(1,25)))
    optargets = optargets.unsqueeze(0).expand(targets.size(0),optargets.size(0))
    Rtargets = Variable(torch.FloatTensor(range(1,num_parts+1)))*24/num_parts   #generating the ending_index of each pre-deifned region
    Rtargets = Rtargets.unsqueeze(0).expand(targets.size(0),Rtargets.size(0))          #expanding over the whole mini-batch
    ratio = ratio.unsqueeze(1).expand_as(Rtargets)                                     #expand ratio
    start_ratio = start_ratio.unsqueeze(1).expand_as(Rtargets)                        #expand start_ratio
    prange = (Rtargets-24*start_ratio.float())/ratio.float()   #64*6                                     #generating the ending_index of each region after crop and resizing operation
    jitter_prange = prange.detach()
    prange = prange.round()
    prange = prange.unsqueeze(1).expand(prange.size(0),24,prange.size(1))              #64*24*6， generating p copies
    tmp = []
    for i in range(0,num_parts):
        tmp.append((optargets.le(prange[:,:,i])))                                      #return 1 if a pixel is within region i or i-1, ..,0
    ptargets = torch.cat([t.unsqueeze(2) for t in tmp],2)                              # pixel 0 is assgined with label p
    ptargets = num_parts-ptargets.sum(2)                                               # inverse the labels 
    jitter = np.random.uniform(-2,2,[jitter_prange.size(0),jitter_prange.size(1)])     # add jitter
    jitter = torch.FloatTensor(jitter)
    jitter_prange += jitter
    jitter_prange = jitter_prange.round()
    jitter_prange = jitter_prange.unsqueeze(1).expand(jitter_prange.size(0),24,jitter_prange.size(1))    #64*24*6
    tmp = []
    for i in range(0,num_parts):
        tmp.append((optargets.le(jitter_prange[:,:,i])))
    jitter_ptargets = torch.cat([t.unsqueeze(2) for t in tmp],2)
    jitter_ptargets = num_parts-jitter_ptargets.sum(2)
    jitter_ptargets = jitter_ptargets.clamp(max=num_parts-1)


#    ptargets = jitter_ptargets
    start_p = ptargets[:,1]     # the first and the last region may contains very few pixels, treat a region as invisible if it contains only 1 pixel  (NOT SURE Whether this is benificial)
    end_p = ptargets[:,-2]
        
#======================== turn the ID label of invisible regions to 751 (will be ignored when deducing the softmax loss===========================================#
    for i, tmp in enumerate(start_p.cpu().data.numpy()):
        if tmp > 0:
            targets[i,0:tmp] = 751
    for i, tmp in enumerate(end_p.cpu().data.numpy()):
        if tmp < num_parts-1:
            targets[i,tmp+1:num_parts] = 751



    return targets, ptargets     #return the self-supervised targets label and region label(ptargets)


if __name__ == '__main__':
    targets = torch.tensor([2,3,4])
    ratio = torch.tensor([0.5, 0.7, 1])
    start_ratio = 1-ratio
    num_parts = 6
    t,p=generate_column_labels(targets, ratio, start_ratio, num_parts)
    print(t)
    print(p)



