from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss, ContrastiveLoss
from .utils.meters import AverageMeter
from .utils import Bar
#import Smooth
from torch.nn import functional as F
from torch.nn import KLDivLoss as KLLoss
from torch.nn import LogSoftmax

class BaseTrainer(object):
    def __init__(self, model, criterion, X=0, Y=0, SMLoss_mode=0, Triplet_margin=0.5, num_instances=8):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.criterion2 = TripletLoss(margin=Triplet_margin).cuda()
        self.indx=X
        self.indy=Y
        self.SML_mode=SMLoss_mode
        self.KLDloss = KLLoss()
        self.logsoft = LogSoftmax()
        self.num_instances =num_instances
    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        TLosses = AverageMeter()
     
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, TLoss, prec1 = self._forward(inputs, targets)
            losses.update(loss.data[0], targets.size(0))
            TLosses.update(TLoss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))
            optimizer.zero_grad()
            torch.autograd.backward([1*loss, 1*TLoss],[torch.ones(1).cuda(), torch.ones(1).cuda()]) 
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | TLoss {N_TLoss:.4f} | | Prec {N_prec:.2f} {N_preca:.2f}' .format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
#                              N_bt=batch_time.val,
                              N_bta=batch_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
                                                          N_TLoss = TLosses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()
#        return TLosses.avg, SMLosses.avg
        

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        index = (targets-751).data.nonzero().squeeze_()
        prob = outputs[1].index_select(0,Variable(index))
        targets = targets.index_select(0,Variable(index))
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
#            loss = LSR.LSRLoss(prob, targets, ratio=0.1)
            loss = self.criterion(prob*1, targets)
            prec, = accuracy(prob.data, targets.data)
            prec = prec[0]
            Tloss,prec2 = self.criterion2(outputs[2].index_select(0,Variable(index)), targets)  # triplet loss
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs[1], targets)
            SMLoss = torch.autograd.Variable(torch.zeros(1))
            if SMLoss.requires_grad == False:
                SMLoss.requires_grad = True
        else:
            raise ValueError("Unsupported loss:", self.criterion)
#        return loss, 1*Tloss, SMLoss, prec, radius
        return loss, Tloss, prec
