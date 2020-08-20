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


class BaseTrainer(object):
    def __init__(self, model, criterion, X, Y, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        weight_p = torch.cat([torch.ones(6),torch.zeros(1)])
        weight_id = torch.cat([torch.ones(751),torch.zeros(1)])
        self.model = model
        self.criterion = criterion
        self.criterion_part = nn.CrossEntropyLoss().cuda()
        self.criterion_ID = nn.CrossEntropyLoss(weight = weight_id).cuda()
        self.indx=X
        self.indy=Y
        self.SML_mode=SMLoss_mode
        self.KLoss = KLDivLoss()

    def train(self, epoch, data_loader, optimizer, print_freq=1, num_parts=6):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        plosses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, rp, targets = self._parse_data(inputs)
            loss, ploss, prec1 = self._forward(inputs, rp, targets, num_parts)
#===================================================================================
            loss = torch.cat([tmp.unsqueeze(1) for tmp in loss],1).squeeze(1)
            losses.update(loss.data[0], targets.size(0))
#            cdegrees.update(cdegree.data[0],targets.size(0))
#            sdegrees.update(sdegree.data[0],targets.size(0))
            precisions.update(prec1, targets.size(0))
            plosses.update(ploss.data[0],targets.size(0))

            optimizer.zero_grad()
            all_loss = loss + [ploss]
            torch.autograd.backward(all_loss, [torch.ones(1).cuda()]*len(all_loss))
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | Part_loss {N_S:.4f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, N_bta=batch_time.avg,
                              N_dt=data_time.val, N_dta=data_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
							  N_S = plosses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()




    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        [imgs, rp], _, pids, _ = inputs # inputs[0] is a list consisting of img_tensor and the retained parts(from 2 to 6)
        inputs = Variable(imgs)
        rp = Variable(rp)
        targets = Variable(pids.cuda())
        return inputs, rp, targets

    def _forward(self, inputs, rp, targets, num_parts):
        outputs = self.model(*[inputs, rp])
        tmp = np.ones(targets.size(0))
        ptargets0 = Variable(torch.LongTensor(0*tmp).cuda()).unsqueeze(1)
        ptargets1 = Variable(torch.LongTensor(1*tmp).cuda()).unsqueeze(1)
        ptargets2 = Variable(torch.LongTensor(2*tmp).cuda()).unsqueeze(1)
        ptargets3 = Variable(torch.LongTensor(3*tmp).cuda()).unsqueeze(1)
        ptargets4 = Variable(torch.LongTensor(4*tmp).cuda()).unsqueeze(1)
        ptargets5 = Variable(torch.LongTensor(5*tmp).cuda()).unsqueeze(1)
        ptargets = torch.cat([ptargets0, ptargets1, ptargets2, ptargets3, ptargets4, ptargets5],1)
        
        targets = torch.cat([targets.unsqueeze(1)]*6,1)
        for i, tmp in enumerate(rp.cpu().data.numpy()):
            if tmp<6:
                ptargets[i,tmp:] = 6 
                targets[i,tmp:] =751




        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss0 = self.criterion_ID(outputs[1][0],targets[:,0])
            loss1 = self.criterion_ID(outputs[1][1],targets[:,1])
            loss2 = self.criterion_ID(outputs[1][2],targets[:,2])
            loss3 = self.criterion_ID(outputs[1][3],targets[:,3])
            loss4 = self.criterion_ID(outputs[1][4],targets[:,4])
            loss5 = self.criterion_ID(outputs[1][5],targets[:,5])
            ploss0 = self.criterion_part(outputs[2][0],ptargets[:,0])
            ploss1 = self.criterion_part(outputs[2][1],ptargets[:,1])
            ploss2 = self.criterion_part(outputs[2][2],ptargets[:,2])
            ploss3 = self.criterion_part(outputs[2][3],ptargets[:,3])
            ploss4 = self.criterion_part(outputs[2][4],ptargets[:,4])
            ploss5 = self.criterion_part(outputs[2][5],ptargets[:,5])
            prec, = accuracy(outputs[1][1].data, targets[:,1].data)
            prec = prec[0]
                        
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss0, loss1, loss2, loss3, loss4, loss5,(ploss0+ploss1+ploss2+ploss3+ploss4+ploss5)/3., prec#,  gap_loss
#        return (loss0+loss1+loss2+loss3+loss4+loss5)/1.,  prec,  gap_loss
