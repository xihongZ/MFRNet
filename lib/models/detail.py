import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from .resnet import Resnet18
import torch
import torch.nn as nn

from torch.nn import BatchNorm2d

from torch.nn import BatchNorm2d


class AB(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=nn.BatchNorm2d):
        super(AB, self).__init__()
        self.s2 = SELayer(inplanes, outplanes, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        x2 = self.s2(x)

        return x2


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class SELayer(nn.Module):
    def __init__(self, channel, outchannel, norm_layer=None):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, outchannel // 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(outchannel // 16)
        self.conv2 = nn.Conv2d(outchannel // 16, channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(channel)

    def forward(self, x):
        b, c, _, _ = x.size()
        pool1 = nn.AdaptiveAvgPool2d(1)
        y = pool1(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y).sigmoid()

        # expand_as把一个tensor变成和函数括号内一样形状的tensor，用法与expand（）类似
        return x * y.expand_as(x)

        # def init_weight(self):
        #     for ly in self.children():
        #         if isinstance(ly, nn.Conv2d):
        #             nn.init.kaiming_normal_(ly.weight, a=1)
        #             if not ly.bias is None: nn.init.constant_(ly.bias, 0)
        #
        # def get_params(self):
        #     wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        #     for name, child in self.named_children():
        #         child_wd_params, child_nowd_params = child.get_params()
        #         if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
        #             lr_mul_wd_params += child_wd_params
        #             lr_mul_nowd_params += child_nowd_params
        #         else:
        #             wd_params += child_wd_params
        #             nowd_params += child_nowd_params
        #     return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
