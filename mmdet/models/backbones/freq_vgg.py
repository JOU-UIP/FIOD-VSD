# vgg.py
# Copyright (c) Open-MMLab. All rights reserved.

import torch.nn as nn
from mmcv.cnn.utils import constant_init, kaiming_init, normal_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmdet.models.builder import BACKBONES
from torch import max as tmax
from torch import mean as tmean
from torch import cat as tcat
import torch_dct
from torchvision.utils import save_image
from torch import ones_like,randn
import torch

def conv3x3(in_planes, out_planes, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation)


def make_vgg_layer(inplanes,
                   planes,
                   num_blocks,
                   dilation=1,
                   with_bn=False,
                   ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


@BACKBONES.register_module()
class Freq_VGG(nn.Module):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """

    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 depth,
                 with_bn=False,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 bn_eval=True,
                 norm_eval=False,
                 bn_frozen=False,
                 ceil_mode=False,
                 with_last_pool=True,
                 spatial_kernel=7,
                 **kwargs):
        super(Freq_VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert 1 <= num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.norm_eval = norm_eval

        self.inplanes = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            planes = 64 * 2 ** i if i < 4 else 512
            vgg_layer = make_vgg_layer(
                self.inplanes,
                planes,
                num_blocks,
                dilation=dilation,
                with_bn=with_bn,
                ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.inplanes = planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                          padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.features:
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

    def forward(self, x):
        # dct
        dct_x=torch_dct.dct_2d(x)
        dct_max_out, _ = tmax(dct_x, dim=1, keepdim=True)
        dct_avg_out = tmean(dct_x, dim=1, keepdim=True)
        mask = self.sigmoid(self.conv(tcat([dct_max_out, dct_avg_out], dim=1)))
        # torch.save(mask,'./work_dirs/save_file/mask.pth')

        # ======random of DVF
        # n_mask = 1 - mask
        # DIF_img_dct = dct_x * mask
        # DVF_img_dct = dct_x * n_mask
        # # refrence=ones_like(x)
        # # noise add to DVF
        # refrence=DVF_img_dct*(randn(n_mask.shape).to(DVF_img_dct))
        # # combination
        # dct_x = refrence + DIF_img_dct
        # idct_x=torch_dct.idct_2d(dct_x)
        # x=x+idct_x
        
        # ======inhibition of DVF
        dct_x = mask * dct_x
        idct_x=torch_dct.idct_2d(dct_x)
        x=x+idct_x

        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        vgg_layers = getattr(self, self.module_name)
        for i in range(self.frozen_stages):
            for j in range(*self.range_sub_modules[i]):
                m = vgg_layers[j]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(Freq_VGG, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
