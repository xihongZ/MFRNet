import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNB1lock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=nn.BatchNorm2d):
        super(CNB1lock, self).__init__()
        self.s2 = H2_Module(inplanes, outplanes, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        _, _, h, w = x.size()
        x2 = self.s2(x)
        x = x2

        return x


class H2_Module(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(H2_Module, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(outplanes)
        self.conv1_2 = nn.Conv2d(outplanes, outplanes, kernel_size=(3, 1), stride=1, padding=(2, 0), dilation=(2, 1))
        # self.depth_conv1_2 = nn.Conv2d(outplanes, outplanes, kernel_size=(3, 1), stride=1, padding=(2, 0), dilation=(2, 1), groups=outplanes)
        self.bn3 = norm_layer(outplanes)
        self.conv2_1 = nn.Conv2d(outplanes, outplanes, kernel_size=(1, 3), stride=1, padding=(0, 2), dilation=(1, 2))
        # self.depth_conv2_1 = nn.Conv2d(outplanes, outplanes, kernel_size=(1, 3), stride=1, padding=(0, 2), dilation=(1, 2), groups=outplanes)
        self.bn4 = norm_layer(outplanes)
        # self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.conv5 = nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn5 = norm_layer(outplanes)

    def forward(self, x):
        _, _, h, w = x.size()

        if torch.is_tensor(h):
            h = h.item()
            w = w.item()
        pool1 = nn.AdaptiveAvgPool2d((h, 1))
        pool2 = nn.AdaptiveAvgPool2d((1, w))

        # x1 = self.pool1(x)
        x1 = pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.conv1_2(x1)
        # x1 = self.depth_conv1_2(x1)
        x1 = self.bn3(x1)
        x1 = x1.expand(-1, -1, h, w)

        # x2 = self.pool2(x)
        x2 = pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.conv2_1(x2)
        # x2 = self.depth_conv2_1(x2)
        x2 = self.bn4(x2)
        x2 = x2.expand(-1, -1, h, w)

        # x3 = self.conv3_1(x)
        x3 = self.conv3(x)
        x3 = self.bn5(x3)
        x3 = x3.expand(-1, -1, h, w)

        x4 = (x1 + x2 + x3)
        x = self.conv5(x)
        x = x * x4
        return x