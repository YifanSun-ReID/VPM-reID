from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss, PartialTripletLoss, ArcFaceLoss, CosFaceLoss
from .utils.meters import AverageMeter
from .utils import Bar
#import Smooth
from torch.nn import functional as F
from torch.nn import KLDivLoss
import numpy as np
import torch.nn as nn
import pdb
from reid.generate_column_labels import generate_column_labels


class BaseTrainer(object):
    def __init__(self, model, criterion, Triplet_margin=1.0):
        super(BaseTrainer, self).__init__()
        weight_id = torch.cat([torch.ones(751),torch.zeros(1)]).cuda()
        weight_part = torch.FloatTensor([1.25, 1.05, 1.15, 1.90])   # The occurance of different part is not equal, assigning different weights might be better, not used yet.
        weight_part = weight_part/weight_part.sum()
        self.model = model
        self.criterion = criterion
       # self.criterion_part = nn.CrossEntropyLoss(weight = weight_part).cuda()
        self.criterion_part = nn.CrossEntropyLoss().cuda()
        self.criterion_ID = nn.CrossEntropyLoss(weight = weight_id).cuda()
        self.criterion_tri = PartialTripletLoss(margin=Triplet_margin).cuda()
        self.criterion_ID_cos = CosFaceLoss(m=0, s=16, weight = weight_id).cuda()
        
       # self.model.eval()    # it may promote cross-domain re-ID by freezing BN parameters of the backboned model
       # self.model.module.drop.train(True)
       # self.model.module.instance.train(True)

    def train(self, epoch, data_loader, optimizer, print_freq=1, num_parts=6):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        plosses = AverageMeter()
        glosses = AverageMeter()
        Tlosses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, ratio, start_ratio, targets = self._parse_data(inputs)
            loss, gloss, ploss, Tloss, prec1 = self._forward(inputs, ratio, start_ratio, targets, num_parts)
            Tloss = Tloss*3

            part_loss = torch.cat([tmp.unsqueeze(0) for tmp in loss], 0).mean()
            losses.update(part_loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))
            plosses.update(ploss.item(),targets.size(0))
            glosses.update(gloss.item(),targets.size(0))
            Tlosses.update(Tloss.item(),targets.size(0))
            optimizer.zero_grad()
            weight_ploss = min((epoch/10)/10., 1.0)     #considering dynamically adjusting the weight of part-classifier loss, not used
            #ploss = ploss*weight_ploss
            total_loss = loss + [gloss, ploss, Tloss]    #please note that loss is also a list
           # total_loss = [gloss, ploss, Tloss]
            torch.autograd.backward(total_loss, [torch.ones(1).squeeze(0).cuda()]*len(total_loss))
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} | Loss {N_loss:.3f} | Ploss {N_S:.4f} | Prec {N_prec:.2f} | GLoss {N_g: .3f} | Tloss {N_tri: .3f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.avg, 
                              N_loss=losses.avg,
				 N_S = plosses.avg,
                              N_prec=precisions.avg,
                              N_g = glosses.avg, N_tri = Tlosses.avg,
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
        targets = Variable(pids.cuda())
        return inputs, ratio, start_ratio, targets

    def _forward(self, inputs, ratio, start_ratio, targets, num_parts):
#        outputs = self.model(*[inputs, ratio])

        targets, ptargets = generate_column_labels(targets, ratio, start_ratio, num_parts)
        outputs = self.model(*[inputs, targets.float()])


        if self.criterion== 'ce':

            loss = []
            for i in range(targets.size(1)):
                loss.append(self.criterion_ID(outputs[1][i], targets[:,i]))
            gloss = loss[-1]
            loss = loss[0:-1]
            ploss = 0
            ploss = self.criterion_part(outputs[2],ptargets)
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec.item()
            tri_feat = outputs[0]
            tri_feat = tri_feat.squeeze(3)
            ##############################
            #       detach  pscore       #
            ##############################
            pscore = outputs[3].detach()
           # tri_feat = outputs[0].view(outputs[0].size(0),-1)
           # tri_feat = tri_feat/tri_feat.norm(2,1).unsqueeze(1).expand_as(tri_feat)
           # Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,6])
            Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,-1], ptargets, pscore)
                        
        # elif isinstance(self.criterion, OIMLoss):
        #     loss, outputs = self.criterion(outputs, targets)
        #     prec, = accuracy(outputs.data, targets.data)
        #     prec = prec[0]
        # elif isinstance(self.criterion, TripletLoss):
        #     loss, prec = self.criterion(outputs, targets)
        elif self.criterion == 'cosface':
            loss = []
            for i in range(targets.size(1)):
                loss.append(self.criterion_ID_cos(outputs[1][i], targets[:,i]))
            gloss = loss[-1]
            loss = loss[0:-1]
            ploss = 0
            ploss = self.criterion_part(outputs[2],ptargets)
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec.item()
            tri_feat = outputs[0]
            tri_feat = tri_feat.squeeze(3)
            ##############################
            #       detach  pscore       #
            ##############################
            pscore = outputs[3].detach()
           # tri_feat = outputs[0].view(outputs[0].size(0),-1)
           # tri_feat = tri_feat/tri_feat.norm(2,1).unsqueeze(1).expand_as(tri_feat)
           # Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,6])
            Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,-1], ptargets, pscore)
        elif self.criterion == 'cosface_ce':
            loss = []
            for i in range(targets.size(1)-1):
                loss.append(self.criterion_ID(outputs[1][i], targets[:,i]))
            gloss = self.criterion_ID_cos(outputs[1][targets.size(1)-1], targets[:,-1])
            ploss = 0
            ploss = self.criterion_part(outputs[2],ptargets)
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec.item()
            tri_feat = outputs[0]
            tri_feat = tri_feat.squeeze(3)
            ##############################
            #       detach  pscore       #
            ##############################
            pscore = outputs[3].detach()
           # tri_feat = outputs[0].view(outputs[0].size(0),-1)
           # tri_feat = tri_feat/tri_feat.norm(2,1).unsqueeze(1).expand_as(tri_feat)
           # Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,6])
            Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,-1], ptargets, pscore)
        elif self.criterion == 'concate_cosface':
            loss = []
            loss.append(self.criterion_ID_cos(outputs[1][0], targets[:,-1]))
            loss.append(self.criterion_ID_cos(outputs[1][1], targets[:,-1]))
            gloss = loss[-1]
            loss = loss[0:-1]
            ploss = 0
            ploss = self.criterion_part(outputs[2],ptargets)
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec.item()
            tri_feat = outputs[0]
            tri_feat = tri_feat.squeeze(3)
            ##############################
            #       detach  pscore       #
            ##############################
            pscore = outputs[3].detach()
           # tri_feat = outputs[0].view(outputs[0].size(0),-1)
           # tri_feat = tri_feat/tri_feat.norm(2,1).unsqueeze(1).expand_as(tri_feat)
           # Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,6])
            Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,-1], ptargets, pscore)

        elif self.criterion == 'concate_cosface_wo_global':
            loss = []
            loss.append(self.criterion_ID_cos(outputs[1][0], targets[:,-1]))
            loss.append(self.criterion_ID_cos(outputs[1][1], targets[:,-1]))
            gloss = 0*loss[-1]
            loss = loss[0:-1]
            ploss = 0
            ploss = self.criterion_part(outputs[2],ptargets)
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec.item()
            tri_feat = outputs[0]
            tri_feat = tri_feat.squeeze(3)
            ##############################
            #       detach  pscore       #
            ##############################
            pscore = outputs[3].detach()
           # tri_feat = outputs[0].view(outputs[0].size(0),-1)
           # tri_feat = tri_feat/tri_feat.norm(2,1).unsqueeze(1).expand_as(tri_feat)
           # Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,6])
            Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,-1], ptargets, pscore)

        elif self.criterion == 'concate_cosface_wo_triplet':
            loss = []
            loss.append(self.criterion_ID_cos(outputs[1][0], targets[:,-1]))
            loss.append(self.criterion_ID_cos(outputs[1][1], targets[:,-1]))
            gloss = loss[-1]
            loss = loss[0:-1]
            ploss = 0
            ploss = self.criterion_part(outputs[2],ptargets)
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec.item()
            tri_feat = outputs[0]
            tri_feat = tri_feat.squeeze(3)
            ##############################
            #       detach  pscore       #
            ##############################
            pscore = outputs[3].detach()
           # tri_feat = outputs[0].view(outputs[0].size(0),-1)
           # tri_feat = tri_feat/tri_feat.norm(2,1).unsqueeze(1).expand_as(tri_feat)
           # Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,6])
            Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,-1], ptargets, pscore)
            Tloss = 0*Tloss

        elif self.criterion == 'concate_cosface_wo_g_t':
            loss = []
            loss.append(self.criterion_ID_cos(outputs[1][0], targets[:,-1]))
            loss.append(self.criterion_ID_cos(outputs[1][1], targets[:,-1]))
            gloss = 0 * loss[-1]
            loss = loss[0:-1]
            ploss = 0
            ploss = self.criterion_part(outputs[2],ptargets)
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec.item()
            tri_feat = outputs[0]
            tri_feat = tri_feat.squeeze(3)
            ##############################
            #       detach  pscore       #
            ##############################
            pscore = outputs[3].detach()
           # tri_feat = outputs[0].view(outputs[0].size(0),-1)
           # tri_feat = tri_feat/tri_feat.norm(2,1).unsqueeze(1).expand_as(tri_feat)
           # Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,6])
            Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,-1], ptargets, pscore)
            Tloss = 0*Tloss

        elif self.criterion == 'concate_cosface_ce':
            loss = []
            loss.append(self.criterion_ID(outputs[1][0], targets[:,-1]))
            loss.append(self.criterion_ID_cos(outputs[1][1], targets[:,-1]))
            gloss = loss[-1]
            loss = loss[0:-1]
            ploss = 0
            ploss = self.criterion_part(outputs[2],ptargets)
            prec, = accuracy(outputs[1][-1].data, targets[:,-1].data)
            prec = prec.item()
            tri_feat = outputs[0]
            tri_feat = tri_feat.squeeze(3)
            ##############################
            #       detach  pscore       #
            ##############################
            pscore = outputs[3].detach()
           # tri_feat = outputs[0].view(outputs[0].size(0),-1)
           # tri_feat = tri_feat/tri_feat.norm(2,1).unsqueeze(1).expand_as(tri_feat)
           # Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,6])
            Tloss, prec2 = self.criterion_tri(tri_feat, targets[:,-1], ptargets, pscore)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, gloss, ploss*0.5, Tloss, prec#,  gap_loss
#        return (loss0+loss1+loss2+loss3+loss4+loss5)/1.,  prec,  gap_loss
