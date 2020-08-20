from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import numpy as np
__all__ = ['resnet50_pseudo_column']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, FCN=False, T=1, dim = 256):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.FCN=FCN
        self.T = T
        self.reduce_dim = dim
#        self.offset = ConvOffset2D(32)
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

#==========================add dilation=============================#
        if self.FCN:
            self.base.layer4[0].conv2.stride=(1,1)
#            self.base.layer4[0].conv2.dilation=(2,2)
#            self.base.layer4[0].conv2.padding=(2,2)
            self.base.layer4[0].downsample[0].stride=(1,1)
#            self.base.layer4[1].conv2.dilation=(2,2)
#            self.base.layer4[1].conv2.padding=(2,2)
#            self.base.layer4[2].conv2.dilation=(2,2)
#            self.base.layer4[2].conv2.padding=(2,2)
#================append conv for FCN==============================#
            self.num_features = num_features
            self.num_classes = num_classes
            self.dropout = dropout
            self.local_conv = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)
            init.kaiming_normal(self.local_conv.weight, mode= 'fan_out')
#            init.constant(self.local_conv.bias,0)
            self.feat_bn2d = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
            init.constant(self.feat_bn2d.weight,1) #initialize BN, may not be used
            init.constant(self.feat_bn2d.bias,0) # iniitialize BN, may not be used
            self.global_conv = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)
            init.kaiming_normal(self.global_conv.weight, mode= 'fan_out')
            self.feat_bn_global = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
            init.constant(self.feat_bn_global.weight,1) #initialize BN, may not be used
            init.constant(self.feat_bn_global.bias,0) # iniitialize BN, may not be used

            self.instance = nn.ModuleList()
            for i in range(7):
                self.instance.append(nn.Linear(self.num_features, self.num_classes))
            for ins in self.instance:
                init.normal_(ins.weight, std=0.001)
                init.constant_(ins.bias, 0)
##---------------------------stripe1----------------------------------------------#

            self.drop = nn.Dropout(self.dropout)
            self.local_mask = nn.Conv2d(self.reduce_dim, 7 , kernel_size=1,padding=0,bias=True)
            init.kaiming_normal(self.local_mask.weight, mode= 'fan_out')
#            init.xavier_normal(self.local_mask.weight)
            init.constant(self.local_mask.bias,0)



#            self.local_maskb = nn.Conv2d(64, 9 , kernel_size=1,padding=0,bias=True)
#            init.kaiming_normal(self.local_maskb.weight, mode= 'fan_out')
#            init.constant(self.local_maskb.bias,0)
#            self.local_mask.requires_grad = False
#            self.local_mask.requires_grad = False
#            self.instance0.requires_grad = False
#            self.instance1.requires_grad = False
#            self.instance2.requires_grad = False
#            self.instance3.requires_grad = False
#            self.instance4.requires_grad = False
#            self.instance5.requires_grad = False
#===================================================================#

        elif not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
#                self.f_bn = nn.BatchNorm1d(2048)
#                init.constant(self.f_bn.weight, 1)
#                init.constant(self.f_bn.bias, 0)

                self.feat = nn.Linear(out_planes, self.num_features, bias=False)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
#                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
#                self.classifier = nn.Linear(self.num_features, self.num_classes)
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

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
#            if name == 'layer4':
#                y = x
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
#            x = F.avg_pool2d(x,x.size()[2:])
#            x = x.view(x.size(0),-1)
#            x = x/ x.norm(2,1).expand_as(x)
            return x
#=======================FCN===============================#
        if self.FCN:
            T = self.T         
            y = self.drop(x).unsqueeze(1)
#            y = (x).unsqueeze(1)
            stride = 2048/self.reduce_dim
#            require_dim = stride*self.reduce_dim
#            y = y[:,0,0:require_dim,:,:] 
            y = F.avg_pool3d(y,kernel_size=(stride,1,8),stride=(stride,1,8)).squeeze(1)
#            y = x[:,0:64] 
#            center = F.avg_pool2d(y,(y.size(2),y.size(3)))
#            y = y-center.expand_as(y)
            x_global = F.avg_pool2d(x, (24,8))
            part_feature = []
            for i in range(6):
                tmp = part_labels.eq(i)
                area = torch.clamp((tmp.sum(1)*8).unsqueeze(1).float(), min=1e-12)
                tmp = (x*tmp.float().unsqueeze(1).unsqueeze(3).expand_as(x)).sum(3).sum(2)/area
                part_feature.append(tmp.unsqueeze(2).unsqueeze(3))
            x = torch.cat(part_feature,2)   
            
            local_score = self.local_mask(y)
            local_score = local_score.squeeze(3)
            
#====================================================================================
#==================================================================================#
            out0 = x.view(x.size(0),-1)
            out0 = x/torch.clamp(x.norm(2,1).unsqueeze(1).expand_as(x),min=1e-12)
            '''
            score = F.softmax(1*local_score,1)
            bb, cc, hh, ww = x.size()
            feat = x.unsqueeze(2).expand(bb,cc, 7, hh, ww)*score.unsqueeze(1).unsqueeze(4).expand(bb,cc,7, hh, ww)
            feat = feat.sum(4).sum(3).unsqueeze(3)
            feat = feat0/torch.clamp((feat.norm(2,1).unsqueeze(1).expand_as(feat)),min=1e-12)
            ''' 
            
            x = self.drop(x)
            x = self.local_conv(x)
            x_global = self.drop(x_global)
            x_global = self.global_conv(x_global)

            out1 = x.view(x.size(0),-1)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            
            x = self.feat_bn2d(x)
            x_global = self.feat_bn_global(x_global) 
#            out1 = x.view(x.size(0),-1)
#            out1 = out1/out1.norm(2,1).unsqueeze(1).expand_as(out1)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            x = F.relu(x) # relu for local_conv feature
            x_global = F.relu(x_global)
            x = x.chunk(6,2)
            x = list(x)
            c =[]
            for i in range(6):
                x[i] = x[i].contiguous().view(x[i].size(0),-1)
                c.append(self.instance[i](x[i]))
            c.append(self.instance[6](x_global.view(x_global.size(0),-1)))
            ps = []
            part_score = local_score.chunk(24,2)
            for i in range(24):
                ps.append(part_score[i].contiguous().view(local_score.size(0),-1))
            return out0, c, ps#, pool5#, orig_weight, pool5#, (, x1, x2, x3, x4, x5) #, glob#, c6, c7

#            return out0, x 
#==========================================================#


        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        out1 = x
        out1 = x / x.norm(2,1).unsqueeze(1).expand_as(x)
#        x = self.drop(x)
        if self.has_embedding:
            x = self.feat(x)
#            out2 = x
#            out2 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
            x = self.feat_bn(x)
            out2 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
        if self.norm:
            x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
#        elif self.has_embedding:  # adding relu after fc
#            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)


        return out2, x
	

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)






def resnet50_pseudo_column(**kwargs):
    return ResNet(50, **kwargs)
