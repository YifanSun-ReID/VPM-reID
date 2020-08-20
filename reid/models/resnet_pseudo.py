from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
__all__ = ['resnet50_pseudo']


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

#            self.local_conv1 = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)
#            init.kaiming_normal(self.local_conv1.weight, mode= 'fan_out')
#            self.feat_bn2d1 = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
#            init.constant(self.feat_bn2d1.weight,1) #initialize BN, may not be used
#            init.constant(self.feat_bn2d1.bias,0) # iniitialize BN, may not be used

#            self.local_conv2 = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)
#            init.kaiming_normal(self.local_conv2.weight, mode= 'fan_out')
#            self.feat_bn2d2 = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
#            init.constant(self.feat_bn2d2.weight,1) #initialize BN, may not be used
#            init.constant(self.feat_bn2d2.bias,0) # iniitialize BN, may not be used

#            self.local_conv3 = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)
#            init.kaiming_normal(self.local_conv3.weight, mode= 'fan_out')
#            self.feat_bn2d3 = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
#            init.constant(self.feat_bn2d3.weight,1) #initialize BN, may not be used
#            init.constant(self.feat_bn2d3.bias,0) # iniitialize BN, may not be used

#            self.local_conv4 = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)
#            init.kaiming_normal(self.local_conv4.weight, mode= 'fan_out')
#            self.feat_bn2d4 = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
#            init.constant(self.feat_bn2d4.weight,1) #initialize BN, may not be used
#            init.constant(self.feat_bn2d4.bias,0) # iniitialize BN, may not be used

#            self.local_conv5 = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)
#            init.kaiming_normal(self.local_conv5.weight, mode= 'fan_out')
#            self.feat_bn2d5 = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
#            init.constant(self.feat_bn2d5.weight,1) #initialize BN, may not be used
#            init.constant(self.feat_bn2d5.bias,0) # iniitialize BN, may not be used



##---------------------------stripe1----------------------------------------------#
            self.instance0 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance0.weight, std=0.001)
            init.constant(self.instance0.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance1 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance1.weight, std=0.001)
            init.constant(self.instance1.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance2 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance2.weight, std=0.001)
            init.constant(self.instance2.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance3 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance3.weight, std=0.001)
            init.constant(self.instance3.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance4 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance4.weight, std=0.001)
            init.constant(self.instance4.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance5 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance5.weight, std=0.001)
            init.constant(self.instance5.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
#            self.instance6 = nn.Linear(self.num_features, self.num_classes)
#            init.normal(self.instance6.weight, std=0.001)
#            init.constant(self.instance6.bias, 0)
##---------------------------stripe1----------------------------------------------#
#            self.instance7 = nn.Linear(self.num_features, self.num_classes)
#            init.normal(self.instance7.weight, std=0.001)
#            init.constant(self.instance7.bias, 0)
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

    def forward(self, inputs, rp=None):
        x = inputs
        if rp is None:
            rp = torch.ones(inputs.size(0))*3
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
            y = F.avg_pool3d(y,kernel_size=(stride,1,1),stride=(stride,1,1)).squeeze(1)
#            y = x[:,0:64] 
#            center = F.avg_pool2d(y,(y.size(2),y.size(3)))
#            y = y-center.expand_as(y)

            rp = rp.cpu().data.numpy()
            yy = list(y.chunk(64,0))
            xx = list(x.chunk(64,0))
        
            batch_pad_y = [torch.zeros(1,y.size(1),1,1).cuda()]
            batch_pad_x = [torch.zeros(1,x.size(1),1,1).cuda()]
            for i in range(y.size(0)):
                yy[i] = F.adaptive_avg_pool2d(yy[i],(rp[i],1))
                xx[i] = F.adaptive_avg_pool2d(xx[i],(rp[i],1))
                if rp[i]<6:
                    yy[i] = torch.cat((yy[i],torch.cat(batch_pad_y*(6-rp[i]),2)),2)
                    xx[i] = torch.cat((xx[i],torch.cat(batch_pad_x*(6-rp[i]),2)),2)

            y = torch.cat(yy,0)
            x = torch.cat(xx,0)
            local_score = self.local_mask(y)
            local_score = local_score.squeeze(3)
            tmp = local_score
#====================================================================================
            '''
            lw = local_mask.chunk(6,1)
            x = x*6
            f0 = x*(lw[0].expand_as(x))
            f1 = x*(lw[1].expand_as(x))
            f2 = x*(lw[2].expand_as(x))
            f3 = x*(lw[3].expand_as(x))
            f4 = x*(lw[4].expand_as(x))
            f5 = x*(lw[5].expand_as(x))
            f0 = F.avg_pool2d(f0,kernel_size=(f0.size(2),f0.size(3)))  
            f1 = F.avg_pool2d(f1,kernel_size=(f1.size(2),f1.size(3)))  
            f2 = F.avg_pool2d(f2,kernel_size=(f2.size(2),f2.size(3)))  
            f3 = F.avg_pool2d(f3,kernel_size=(f3.size(2),f3.size(3)))  
            f4 = F.avg_pool2d(f4,kernel_size=(f4.size(2),f4.size(3)))  
            f5 = F.avg_pool2d(f5,kernel_size=(f5.size(2),f5.size(3))) 
            x = torch.cat((f0,f1,f2,f3,f4,f5),2)
            feat = torch.cat((f0,f1,f2,f3,f4,f5),2)
            
            out0 = feat.view(feat.size(0),-1)
            out0 = out0/out0.norm(2,1).unsqueeze(1).expand_as(out0)
            out0 = feat/feat.norm(2,1).unsqueeze(1).expand_as(feat)
#            out0 = F.max_pool2d(feat,(6,1))
#            out0 = out0/out0.norm(2,1).unsqueeze(1).expand_as(out0)
            '''
#==================================================================================#
            out0 = x.view(x.size(0),-1)
            out0 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            
            x = self.drop(x)
            x = self.local_conv(x)

            out1 = x.view(x.size(0),-1)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            
            x = self.feat_bn2d(x)
#            out1 = x.view(x.size(0),-1)
#            out1 = out1/out1.norm(2,1).unsqueeze(1).expand_as(out1)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            x = F.relu(x) # relu for local_conv feature
            x = x.chunk(6,2)
            x0 = x[0].contiguous().view(x[0].size(0),-1)
            x1 = x[1].contiguous().view(x[1].size(0),-1)
            x2 = x[2].contiguous().view(x[2].size(0),-1)
            x3 = x[3].contiguous().view(x[3].size(0),-1)
            x4 = x[4].contiguous().view(x[4].size(0),-1)
            x5 = x[5].contiguous().view(x[5].size(0),-1)
            c0 = self.instance0(x0)
            c1 = self.instance1(x1)
            c2 = self.instance2(x2)
            c3 = self.instance3(x3)
            c4 = self.instance4(x4)
            c5 = self.instance5(x5)
            part_score = local_score.chunk(6,2)
            ps0 = part_score[0].contiguous().view(part_score[0].size(0),-1)
            ps1 = part_score[1].contiguous().view(part_score[1].size(0),-1)
            ps2 = part_score[2].contiguous().view(part_score[2].size(0),-1)
            ps3 = part_score[3].contiguous().view(part_score[3].size(0),-1)
            ps4 = part_score[4].contiguous().view(part_score[4].size(0),-1)
            ps5 = part_score[5].contiguous().view(part_score[5].size(0),-1)
#            c6 = self.instance6(x[4].contiguous().view(x[6].size()[0],-1))
#            c7 = self.instance5(x[7].contiguous().view(x[7].size()[0],-1))
            return out0, (c0, c1, c2, c3, c4, c5), (ps0, ps1, ps2, ps3, ps4, ps5), tmp#, pool5#, orig_weight, pool5#, (x0, x1, x2, x3, x4, x5) #, glob#, c6, c7

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






def resnet50_pseudo(**kwargs):
    return ResNet(50, **kwargs)
