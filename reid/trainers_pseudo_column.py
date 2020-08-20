from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
#from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter
from .utils import Bar
#import Smooth
from torch.nn import functional as F
from torch.nn import KLDivLoss
import numpy as np
import torch.nn as nn
from reid.generate_column_labels import generate_column_labels
import pdb


class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        weight_id = torch.cat([torch.ones(751),torch.zeros(1)])
        weight_part = torch.FloatTensor([1.5,1,0.5,0.5,1,1.5])
        weight_part = torch.FloatTensor([1.,1.,1.,1.,1.,1.])
        self.model = model
        self.criterion = criterion
        self.criterion_part = nn.CrossEntropyLoss().cuda()
        self.criterion_ID = nn.CrossEntropyLoss(weight = weight_id).cuda()
        self.indx=X
        self.indy=Y
        self.SML_mode=SMLoss_mode
        self.KLoss = KLDivLoss()

#        self.model.eval()
#        self.model.module.drop.train(True)
#        self.model.module.instance.train(True)


    def train(self, epoch, data_loader, optimizer, print_freq=1, num_parts=6):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        plosses = AverageMeter()
        glosses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, ratio, start_ratio, targets = self._parse_data(inputs)
            loss, gloss, ploss, prec1 = self._forward(inputs, ratio, start_ratio, targets, num_parts)
#===================================================================================
            part_loss = torch.cat([tmp.unsqueeze(0) for tmp in loss], 0).mean()
            losses.update(part_loss.data[0], targets.size(0))
#            cdegrees.update(cdegree.data[0],targets.size(0))
#            sdegrees.update(sdegree.data[0],targets.size(0))
            precisions.update(prec1, targets.size(0))
            plosses.update(ploss.data[0],targets.size(0))
            glosses.update(gloss.data[0],targets.size(0))
            optimizer.zero_grad()
            total_loss = loss+[gloss,ploss]
            torch.autograd.backward(total_loss, [torch.ones(1).cuda()]*len(total_loss))
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Part_loss {N_S:.4f} | Prec {N_prec:.2f} {N_preca:.2f}, Global_Loss {N_g: .3f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, 
                              N_loss=losses.val, N_lossa=losses.avg,
							  N_S = plosses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
                              N_g = glosses.avg,
							  )
            bar.next()
        bar.finish()




    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        [imgs, (ratio, start_ratio)], _, pids, _ = inputs # inputs[0] is a list consisting of img_tensor and the retained parts(from 2 to 6)
        inputs = Variable(imgs)
        ratio = Variable(ratio)
        start_ratio = Variable(start_ratio)
        targets = Variable(pids.cuda())
        return inputs, ratio, start_ratio, targets

    def _forward(self, inputs, ratio, start_ratio, targets, num_parts):
#        outputs = self.model(*[inputs, ratio])

        targets, ptargets = generate_column_labels(targets, ratio, start_ratio, num_parts)
        outputs = self.model(*[inputs, ptargets.float()])
        pdb.set_trace()

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = []
            for i in range(targets.size(1)):                                                   # (p+1) branches of ID softmax losses in total, the last one for global feature
                loss.append(self.criterion_ID(outputs[1][i], targets[:,i]))
            gloss = loss[-1]
            loss = loss[0:-1]
            ploss = 0
            for i in range(24):                                                                # Region classifier for each column   (if only vertical crop, we predict one shared result for pixels in a same column
                ploss += self.criterion_part(outputs[2][i],ptargets[:,i])
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec[0]
                        
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, gloss, ploss/12., prec#,  gap_loss
#        return (loss0+loss1+loss2+loss3+loss4+loss5)/1.,  prec,  gap_loss
