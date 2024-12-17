from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch import nn
from ..builder import HEADS
import numpy as np

class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)


class DAPerHead(nn.Module):

    def __init__(self, in_channels, domain_num_classes):

        super(DAPerHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.conv3_da = nn.Conv2d(128, domain_num_classes, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        torch.nn.init.normal_(self.conv3_da.weight, std=0.05)
        torch.nn.init.constant_(self.conv3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1_da(x))
        x=F.dropout(x,p=0.5,training=self.training)
        x=F.relu(self.conv2_da(x))
        x=F.dropout(x,p=0.5,training=self.training)
        x=self.conv3_da(x)
        return x

@HEADS.register_module()
class DAHead(torch.nn.Module):

    def __init__(self,domain_num_classes,in_channels=256):
        super(DAHead, self).__init__()

        self.weight = 0.1
        self.cst_weight = 0.1
        # self.grl_weight=0.1
        self.domain_num_classes=domain_num_classes
               
        self.head = DAPerHead(in_channels,domain_num_classes)
        self.consist_arch_settings =[4,3,2,1]
        for stage,arch in enumerate(self.consist_arch_settings):
            for d in range(arch):
                self.add_module(f'consist_layer{stage}_{d}', nn.Conv2d(in_channels=domain_num_classes,out_channels=domain_num_classes,kernel_size=3,stride=2,padding=1,bias=False))
        self.add_module('consist_layer4_0', nn.Conv2d(in_channels=domain_num_classes,out_channels=domain_num_classes,kernel_size=1,bias=False))

    def consistency_loss(self,da_consist_features,size_average=True):
        losses=[]
        align_da_consist_features=[]
        min_feature=da_consist_features[-1]
        
        for stage,feature in enumerate(da_consist_features[:-1]):
            for d in range(self.consist_arch_settings[stage]):
                consist_conv = getattr(self, f'consist_layer{stage}_{d}')
                feature=consist_conv(feature)
            assert feature.shape==min_feature.shape
            feature = F.relu(feature)        #new idea not try yet!
            feature=F.dropout(feature,p=0.5,training=self.training)
            feature=feature.permute(0, 2, 3, 1)
            align_da_consist_features.append(feature)
        min_feature=self.consist_layer4_0(min_feature)
        min_feature=min_feature.permute(0, 2, 3, 1)
        min_feature = F.relu(min_feature)
        min_feature=F.dropout(min_feature,p=0.5,training=self.training)
        align_da_consist_features.append(min_feature)

        for i in range(len(align_da_consist_features)):
            for j in range(i+1,len(align_da_consist_features)):
                loss=F.cross_entropy(align_da_consist_features[i].reshape(-1,self.domain_num_classes).softmax(dim=1), 
                                     align_da_consist_features[j].reshape(-1,self.domain_num_classes).softmax(dim=1))
                loss=loss.unsqueeze(0)
                losses.append(loss)
        losses=torch.cat(losses)

        if size_average:
            return losses.mean()
        return losses.sum()

    def loss(self, da_features, da_consist_features, da_labels):

        da_flattened = []
        da_labels_flattened = []

        for da_per_level in da_features:
            N, A, H, W = da_per_level.shape
            da_per_level = da_per_level.permute(0, 2, 3, 1)
            da_label_per_level = torch.zeros_like(da_per_level, dtype=torch.float32)
            for i,idx in enumerate(da_labels):
                da_label_per_level[i,:,:,idx]=1
            
            da_per_level = da_per_level.reshape(N, -1, A)
            da_label_per_level = da_label_per_level.reshape(N, -1, A)
            
            da_flattened.append(da_per_level)
            da_labels_flattened.append(da_label_per_level)
            
        da_flattened = torch.cat(da_flattened, dim=1).reshape(-1,self.domain_num_classes).softmax(dim=1)
        da_labels_flattened = torch.cat(da_labels_flattened, dim=1).reshape(-1,self.domain_num_classes).softmax(dim=1)
        
        da_loss = F.cross_entropy(
            da_flattened, da_labels_flattened)
        # import random
        # n=random.randint(1,100)
        # torch.save(da_consist_features[0],f'./work_dirs/save_image/{n}_label{da_labels[0][0]}.pth')
        # torch.save(da_consist_features[1],f'./work_dirs/save_image/{n}_label{da_labels[1][0]}.pth')
        da_consist_loss = self.consistency_loss(da_consist_features)

        return da_loss, da_consist_loss

    def forward_step(self,feature):

        grl_feature=grad_reverse(feature)
        da_feature=self.head(grl_feature)
        da_consist_feature=self.head(feature)
        return da_feature,da_consist_feature

    def forward(self, x, gt_domain_labels):

        da_features=[]
        da_consist_features=[]
        #single head
        # da_feature,da_consist_feature=self.forward_step(x[-1])
        # da_features.append(da_feature)
        # da_consist_features.append(da_consist_feature)
        #multi head
        for feature in x:
            da_feature,da_consist_feature=self.forward_step(feature)
            da_features.append(da_feature)
            da_consist_features.append(da_consist_feature)
        
        if self.training:
            da_loss, da_consistency_loss = self.loss(
                da_features, da_consist_features, gt_domain_labels)
            losses = {}
            if self.weight > 0:
                losses["loss_da"] = self.weight * da_loss
            if self.cst_weight > 0:
                losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
            return losses
        return {}
    
@HEADS.register_module()
class  Mult_DAHead(torch.nn.Module):

    def __init__(self,domain_num_classes,in_channels=256):
        super( Mult_DAHead, self).__init__()

        self.weight = 0.1
        self.cst_weight = 0.1
        # self.grl_weight=0.1
        self.domain_num_classes=domain_num_classes
               
        # self.head = DAPerHead(in_channels,domain_num_classes)
        for i in range(5):
            setattr(self, f'head{i}', DAPerHead(in_channels,domain_num_classes))
        self.consist_arch_settings =[4,3,2,1]
        for stage,arch in enumerate(self.consist_arch_settings):
            for d in range(arch):
                self.add_module(f'consist_layer{stage}_{d}', nn.Conv2d(in_channels=domain_num_classes,out_channels=domain_num_classes,kernel_size=3,stride=2,padding=1,bias=False))
        self.add_module('consist_layer4_0', nn.Conv2d(in_channels=domain_num_classes,out_channels=domain_num_classes,kernel_size=1,bias=False))

    def consistency_loss(self,da_consist_features,size_average=True):
        losses=[]
        align_da_consist_features=[]
        min_feature=da_consist_features[-1]
        
        for stage,feature in enumerate(da_consist_features[:-1]):
            for d in range(self.consist_arch_settings[stage]):
                consist_conv = getattr(self, f'consist_layer{stage}_{d}')
                feature=consist_conv(feature)
            assert feature.shape==min_feature.shape
            feature=feature.permute(0, 2, 3, 1)
            align_da_consist_features.append(feature)
        min_feature=min_feature.permute(0, 2, 3, 1)
        align_da_consist_features.append(min_feature)

        for i in range(len(align_da_consist_features)):
            for j in range(i+1,len(align_da_consist_features)):
                loss=F.cross_entropy(align_da_consist_features[i].reshape(-1,self.domain_num_classes).softmax(dim=1), 
                                     align_da_consist_features[j].reshape(-1,self.domain_num_classes).softmax(dim=1))
                loss=loss.unsqueeze(0)
                losses.append(loss)
        losses=torch.cat(losses)

        if size_average:
            return losses.mean()
        return losses.sum()

    def loss(self, da_features, da_consist_features, da_labels):

        da_flattened = []
        da_labels_flattened = []

        for da_per_level in da_features:
            N, A, H, W = da_per_level.shape
            da_per_level = da_per_level.permute(0, 2, 3, 1)
            da_label_per_level = torch.zeros_like(da_per_level, dtype=torch.float32)
            for i,idx in enumerate(da_labels):
                da_label_per_level[i,:,:,idx]=1
            
            da_per_level = da_per_level.reshape(N, -1, A)
            da_label_per_level = da_label_per_level.reshape(N, -1, A)
            
            da_flattened.append(da_per_level)
            da_labels_flattened.append(da_label_per_level)
            
        da_flattened = torch.cat(da_flattened, dim=1).reshape(-1,self.domain_num_classes).softmax(dim=1)
        da_labels_flattened = torch.cat(da_labels_flattened, dim=1).reshape(-1,self.domain_num_classes).softmax(dim=1)
        
        da_loss = F.cross_entropy(
            da_flattened, da_labels_flattened)
        # import random
        # n=random.randint(1,100)
        # torch.save(da_consist_features[0],f'./work_dirs/save_image/{n}_label{da_labels[0][0]}.pth')
        # torch.save(da_consist_features[1],f'./work_dirs/save_image/{n}_label{da_labels[1][0]}.pth')
        da_consist_loss = self.consistency_loss(da_consist_features)

        return da_loss, da_consist_loss


    def forward_step(self,feature,i):

        grl_feature=grad_reverse(feature)
        da_feature=getattr(self,'head'+str(i))(grl_feature)
        da_consist_feature=getattr(self,'head'+str(i))(feature)
        return da_feature,da_consist_feature

    def forward(self, x, gt_domain_labels):

        da_features=[]
        da_consist_features=[]
        for i,feature in enumerate(x):
            da_feature,da_consist_feature=self.forward_step(feature,i)
            da_features.append(da_feature)
            da_consist_features.append(da_consist_feature)
        
        if self.training:
            da_loss, da_consistency_loss = self.loss(
                da_features, da_consist_features, gt_domain_labels)
            losses = {}
            if self.weight > 0:
                losses["loss_da"] = self.weight * da_loss
            if self.cst_weight > 0:
                losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
            return losses
        return {}