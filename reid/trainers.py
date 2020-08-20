from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils import Bar
#import Smooth
from torch.nn import functional as F
from torch.nn import KLDivLoss as KLLoss
from torch.nn import LogSoftmax

class BaseTrainer(object):
    def __init__(self, model, criterion, X=0, Y=0, SMLoss_mode=0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.indx=X
        self.indy=Y
        self.SML_mode=SMLoss_mode
        self.KLDloss = KLLoss()
        self.logsoft = LogSoftmax()
    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        SMLosses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(data_loader))
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, SMLoss, prec1 = self._forward(inputs, targets)
            losses.update(loss.data[0], targets.size(0))
            SMLosses.update(SMLoss.data[0],targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            torch.autograd.backward([loss,SMLoss],[torch.ones(1).cuda(), torch.ones(1).cuda()]) 
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = 'Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bt:.3f} {N_bta:.3f} | Data {N_dt:.3f} {N_dta:.3f} | Loss {N_loss:.3f} {N_lossa:.3f} | SMLoss {N_SMLoss:.4f} | Prec {N_prec:.2f} {N_preca:.2f}'.format(
                      N_epoch=epoch, N_batch=i + 1, N_size=len(data_loader),
                              N_bt=batch_time.val, N_bta=batch_time.avg,
                              N_dt=data_time.val, N_dta=data_time.avg,
                              N_loss=losses.val, N_lossa=losses.avg,
							  N_SMLoss=SMLosses.avg,
                              N_prec=precisions.val, N_preca=precisions.avg,
							  )
            bar.next()
        bar.finish()



    def SMLoss_cos(self, feature):
        if self.SML_mode==0:    # no SMLoss
            return feature[1,1]*0
        elif self.SML_mode==1:   # decorrelate features mode
            dim=feature.size(1)
            co=feature.t().mm(feature)# covariance along different dimension within a minibatch
            l=co.diag().resize(dim,1).sqrt()
            co=co.div(l.mm(l.t())) # normalize the covariance matrix to get similarity matrix
            S = co-co.diag().diag()
            smloss= S.pow(2).sum()/dim
            return smloss*20.0
        elif self.SML_mode==2:
            indx=self.indx
            indy=self.indy
        
            batch_size=feature.size(0)
            le=batch_size**2
            co=feature.mm(feature.t()) # covariance along different samples, please note the difference from mode1
#            min_co=co.min().resize(1,1)
#            min_co.detach_()
#            dump=Variable(torch.ones(batch_size,1).cuda(),requires_grad=False)
#            co=co-dump.mm(min_co).mm(dump.t())
            l=co.diag().resize(batch_size,1).sqrt()
            S=co.div(l.mm(l.t())) # normalize the covariance matrix to get similarity matrix
            torch.nn.functional.relu(S,inplace=True)
            W=S.resize(le,1)
            P=S.div(S.sum(1).unsqueeze(1).mm(torch.autograd.Variable(torch.ones(1,batch_size)).cuda()))
            VP=P.resize(le,1)
            VPT=torch.cat(P.t().unsqueeze(2),0)
            tmp=VP.mm(VPT.t())
#            tmp.detach_()
#            tmp.require_grad = False
            PW=W.index_select(0,indx)
            QW=W.index_select(0,indy)
            K=PW-QW
            K=K**2
            K = K.squeeze(1)
#    smloss=tmp.resize(le*le,1).dot(K)/le+S.min()
            smloss = tmp.resize(le*le).dot(K)/le  #+ S.min()/100
            return 10*smloss
        elif self.SML_mode==3:
            indx=self.indx
            indy=self.indy
        
            batch_size=feature.size(0)
            le=batch_size**2
            feat_len=feature.size(1)
            y=feature.div((((feature**2).sum(1)).sqrt()+Variable(torch.rand(batch_size,1).cuda()*1e-30)).mm(torch.autograd.Variable(torch.ones(1,feat_len)).cuda()))
            dist=((y.unsqueeze(0).expand(batch_size,batch_size,feat_len)-y.unsqueeze(1).expand(batch_size,batch_size,feat_len))**2).sum(2).resize(batch_size,batch_size)
#            dist=dist.sqrt()
            P=torch.nn.functional.softmax(-0.1*dist)
            W=dist.resize(le,1)
            VP=P.resize(le,1)
            VPT=torch.cat(P.t().unsqueeze(2),0)
            tmp=VP.mm(VPT.t())
            PW=W.index_select(0,indx)
            QW=W.index_select(0,indy)
            K=PW-QW
            K=K**2
            smloss=tmp.resize(le*le,1).dot(K)/le
            return 10*smloss

    def gap_loss(self,feature1,feature2):
        batch_size = feature1.size(0)
        v1 = feature1.norm(2,1)
        v2 = feature2.norm(2,1)
        co = ((feature1*feature2).sum(1)/v1/v2).sum()/batch_size
        co = F.relu(co-0.2)
        return co



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
#            SMLoss = self.KLDloss(self.logsoft(outputs[1]),PT)*751
#            SMLoss = self.SMLoss_cos(outputs[0])
            SMLoss = torch.autograd.Variable(torch.zeros(1))
            if SMLoss.requires_grad == False:
                SMLoss.requires_grad = True
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
        return loss,SMLoss, prec
