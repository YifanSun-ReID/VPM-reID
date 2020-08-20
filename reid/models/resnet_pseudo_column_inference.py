from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import numpy as np
import pdb
__all__ = ['resnet50_pseudo_column_inference']


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
            self.num_classes = 752 #num_classes
            self.dropout = dropout
            self.local_conv = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)
            init.kaiming_normal(self.local_conv.weight, mode= 'fan_out')
#            init.constant(self.local_conv.bias,0)
            self.feat_bn2d = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
            init.constant(self.feat_bn2d.weight,1) #initialize BN, may not be used
            init.constant(self.feat_bn2d.bias,0) # iniitialize BN, may not be used


            self.instance = nn.ModuleList()
            for i in range(6):
                self.instance.append(nn.Linear(self.num_features, self.num_classes))
            for ins in self.instance:
                init.normal_(ins.weight, std = 0.001)
                init.constant_(ins.bias, 0)

            self.drop = nn.Dropout(self.dropout)
            self.local_mask = nn.Conv2d(self.reduce_dim, 6 , kernel_size=1,padding=0,bias=True)
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

    def forward(self, inputs, ratio=None):
        x = inputs
        if ratio is None:
            ratio = torch.ones(inputs.size(0))*1
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

            stride = (4/ratio).round().int()
            stride = stride.cpu().data.numpy()
#            yy = list(y.chunk(y.size(0),0))
            xx = list(x.chunk(x.size(0),0))
            zz = F.avg_pool2d(x,(24,8)) 

            x = torch.cat(xx,0)
            local_score = self.local_mask(y)
            local_score = local_score.squeeze(3)
            score = F.softmax(1*local_score,1)
            pscore = score.sum(2)
            thresh = pscore[:,0:2].mean().item()*0.3
            bb, cc, hh, ww = x.size()
            feat = x.unsqueeze(2).expand(bb, cc, 6, hh, ww)*score.unsqueeze(1).unsqueeze(4).expand(bb, cc, 6, hh, ww)
            feat = feat.sum(4).sum(3)
            #===================get the number of effective parts==========
            num_parts = []
            for i in range(6):
                num_parts.append(pscore[:,i].gt(thresh).unsqueeze(1)*i)  # get the index for each activated part
            num_parts = torch.cat(num_parts,1)
            num_parts = num_parts.max(1)[0].cuda()   # the index of the last activated part indicates the number of full parts
#            print num_parts.float().mean().item()

            feat = feat.unsqueeze(3)
            #===================done============================
            x = feat[:,:,0:6]

#====================================================================================
#==================================================================================#
#            out0 = x.view(x.size(0),-1)
            out0 = feat/torch.clamp(feat.norm(2,1).unsqueeze(1).expand_as(feat), min=1e-12)
            out0 = out0[:,:,0:4]
#            out0[:,:,5] = out0[:,:,5]*0.7 
            zz = zz/zz.norm(2,1).unsqueeze(1).expand_as(zz)
#            out0 = torch.cat([out0,zz],2)
            x = self.drop(x)
            x = self.local_conv(x)

            out1 = x.view(x.size(0),-1)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            
            x = self.feat_bn2d(x)
#            out1 = x.view(x.size(0),-1)
#            out1 = out1/out1.norm(2,1).unsqueeze(1).expand_as(out1)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            x = x.chunk(6,2)
            x = list(x)
            c = []
            for i in range(6):
                x[i] = x[i].contiguous().view(x[i].size(0),-1)
                c.append(self.instance[i](x[i]))
            part_score = local_score.chunk(24,2)
            ps = []
            for i in range(24):
                ps.append(part_score[i].contiguous().view(local_score.size(0),-1))

            return out0, c, ps 

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






def resnet50_pseudo_column_inference(**kwargs):
    return ResNet(50, **kwargs)
