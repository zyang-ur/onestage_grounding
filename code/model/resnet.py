#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        leaky=False,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
            ),
        )

        if leaky:
            self.add_module("relu", nn.LeakyReLU(0.1))
        elif relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _Bottleneck(nn.Sequential):
    """Bottleneck Unit"""

    def __init__(
        self, in_channels, mid_channels, out_channels, stride, dilation, downsample
    ):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReLU(in_channels, mid_channels, 1, stride, 0, 1)
        self.conv3x3 = _ConvBatchNormReLU(
            mid_channels, mid_channels, 3, 1, dilation, dilation
        )
        self.increase = _ConvBatchNormReLU(
            mid_channels, out_channels, 1, 1, 0, 1, relu=False
        )
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReLU(
                in_channels, out_channels, 1, stride, 0, 1, relu=False
            )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _ResBlock(nn.Sequential):
    """Residual Block"""

    def __init__(
        self, n_layers, in_channels, mid_channels, out_channels, stride, dilation
    ):
        super(_ResBlock, self).__init__()
        self.add_module(
            "block1",
            _Bottleneck(
                in_channels, mid_channels, out_channels, stride, dilation, True
            ),
        )
        for i in range(2, n_layers + 1):
            self.add_module(
                "block" + str(i),
                _Bottleneck(
                    out_channels, mid_channels, out_channels, 1, dilation, False
                ),
            )

    def __call__(self, x):
        return super(_ResBlock, self).forward(x)


class _ResBlockMG(nn.Sequential):
    """3x Residual Block with multi-grid"""

    def __init__(
        self,
        n_layers,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation,
        mg=[1, 2, 1],
    ):
        super(_ResBlockMG, self).__init__()
        self.add_module(
            "block1",
            _Bottleneck(
                in_channels, mid_channels, out_channels, stride, dilation * mg[0], True
            ),
        )
        self.add_module(
            "block2",
            _Bottleneck(
                out_channels, mid_channels, out_channels, 1, dilation * mg[1], False
            ),
        )
        self.add_module(
            "block3",
            _Bottleneck(
                out_channels, mid_channels, out_channels, 1, dilation * mg[2], False
            ),
        )

    def __call__(self, x):
        return super(_ResBlockMG, self).forward(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out