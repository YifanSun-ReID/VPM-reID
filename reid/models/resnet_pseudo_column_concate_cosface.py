from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Parameter
import torchvision
import torch
import numpy as np
import pdb
__all__ = ['resnet50_pseudo_column_concate_cosface']

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5) # ?????????????

    def forward(self, input):
        x = input   # size=(B,F)  B is batchsize, F is feature len
        w = self.weight # size=(Classnum, F) 

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, FCN=False, T=1, dim = 256, num_parts=6):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.FCN=FCN
        self.T = T
        self.reduce_dim = dim
        self.num_parts = num_parts
        # self.offset = ConvOffset2D(32)
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        # add dilation
        if self.FCN:
            self.base.layer4[0].conv2.stride=(1,1)
            self.base.layer4[0].downsample[0].stride=(1,1)
            # append conv for FCN
            self.num_features = num_features
            self.num_classes = num_classes
            self.dropout = dropout
            self.instance = nn.ModuleList()

            for i in range(self.num_parts+1):
                local_conv = nn.Linear(2048,self.num_features,bias=False)
                init.kaiming_normal_(local_conv.weight, mode='fan_out')

                local_bn = nn.BatchNorm1d(self.num_features)
                init.constant_(local_bn.weight,1)
                init.constant_(local_bn.bias,0)

                # fc = AngleLinear(self.num_features, self.num_classes) 

                self.instance.append(
                    nn.Sequential(
                        nn.Dropout(self.dropout),
                        local_conv,
                        local_bn,
                        #nn.ReLU(inplace=True),
                        #nn.Dropout(self.dropout),
                        # fc
                        )
                    )
            self.global_classifier = AngleLinear(self.num_features, self.num_classes)
            self.part_classifier = AngleLinear(self.num_features*self.num_parts, self.num_classes)

            # stripe1
            self.drop = nn.Dropout(self.dropout)
            self.local_mask = nn.Conv2d(self.reduce_dim, self.num_parts, kernel_size=1, bias=True)
            init.kaiming_normal_(self.local_mask.weight, mode='fan_out')
            init.constant_(self.local_mask.bias, 0)

        elif not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
               # self.f_bn = nn.BatchNorm1d(2048)
               # init.constant_(self.f_bn.weight, 1)
               # init.constant_(self.f_bn.bias, 0)

                self.feat = nn.Linear(out_planes, self.num_features, bias=False)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
               # init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
               # self.classifier = nn.Linear(self.num_features, self.num_classes)
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, inputs, part_labels=None):
        x = inputs
        if part_labels is None:
            tmp = torch.FloatTensor(range(1,25))
            tmp = ((tmp-0.1)/4).int()
            part_labels = tmp.unsqueeze(0).expand(inputs.size(0),tmp.size(0))
            part_labels = torch.autograd.Variable(part_labels.cuda())
        for name, module in self.base._modules.items():
           # if name == 'layer4':
           #     y = x
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
           # x = F.avg_pool2d(x,x.size()[2:])
           # x = x.view(x.size(0),-1)
           # x = x/ x.norm(2,1).expand_as(x)
            return x
        # FCN
        if self.FCN:
            T = self.T         
            y = self.drop(x).unsqueeze(1)
            stride = 2048//self.reduce_dim
            y = F.avg_pool3d(y,kernel_size=(stride,1,8),stride=(stride,1,8)).squeeze(1)


            x_global = F.avg_pool2d(x, (24,8))
            local_score = self.local_mask(y)    #b*6*24*1
            local_score = local_score.squeeze(3)
            
            score = F.softmax(1*local_score.detach(),1)  #b*6*24
            pscore = score.sum(2)     # b*6--->b*6*24
            score = score/pscore.unsqueeze(2).expand_as(score)
            bb, cc, hh, ww = x.size()
            feat = x.unsqueeze(2).expand(bb,cc, self.num_parts, hh, ww)*score.unsqueeze(1).unsqueeze(4).expand(bb,cc,self.num_parts, hh, ww)
            feat = feat.sum(4).sum(3).unsqueeze(3)
            x = feat 

            out0 = x.view(x.size(0),-1)
            out0 = x/torch.clamp(x.norm(2,1).unsqueeze(1).expand_as(x),min=1e-12)
            
            x_list = list(x.chunk(x.size(2),2))
            x_list.append(x_global)
            f = []
            for tensor, branch in zip(x_list, self.instance):
                tensor = tensor.contiguous().view(tensor.size(0),-1)
                f.append(branch(tensor))
            c = []
            c.append(self.part_classifier(torch.cat(f[0:-1], dim=1)))
            c.append(self.global_classifier(f[-1]))

            ps = local_score
            return out0, c, ps, pscore#, pool5#, orig_weight, pool5#, (, x1, x2, x3, x4, x5) #, glob#, c6, c7

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        out1 = x
        out1 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
       # x = self.drop(x)
        if self.has_embedding:
            x = self.feat(x)
           # out2 = x
           # out2 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
            x = self.feat_bn(x)
            out2 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
        if self.norm:
            x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)


        return out2, x
	

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet50_pseudo_column_concate_cosface(**kwargs):
    return ResNet(50, **kwargs)
