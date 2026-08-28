"""Vendored LaDeDa architecture from RAID (pralab/RAID).

Source: https://github.com/pralab/RAID/blob/main/src/external/cavia2024/networks/LaDeDa.py
Used by the cavia2024 deepfake detector. Only LaDeDa9 is exposed since that is
the variant paired with the aimagelab/RAID_ckpt cavia2024 checkpoint.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out = out + residual
        out = self.relu(out)
        return out


class LaDeDa(nn.Module):
    def __init__(self, block, layers, strides=(1, 2, 2, 2), kernel3=(0, 0, 0, 0),
                 preprocess_type="NPR", num_classes=1, pool=True):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=strides[0], kernel3=kernel3[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3])
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.pool = pool
        self.block = block
        self.preprocess_type = preprocess_type
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))
        return nn.Sequential(*layers)

    def interpolate(self, img, factor):
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode="nearest", recompute_scale_factor=True),
            scale_factor=1 / factor, mode="nearest", recompute_scale_factor=True,
        )

    def preprocess(self, x, grad_type):
        if grad_type == "raw":
            return x
        if grad_type == "NPR":
            return x - self.interpolate(x, 0.5)
        raise ValueError(f"Unsupported preprocess_type: {grad_type}")

    def forward(self, x):
        x = self.preprocess(x, self.preprocess_type)
        x = self.conv1(x * 2.0 / 3.0)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)
            x = x.permute(0, 3, 1, 2)
        return x


def LaDeDa9(strides=(2, 2, 2, 1), preprocess_type="NPR", **kwargs):
    return LaDeDa(Bottleneck, [3, 4, 6, 3], strides=strides,
                  kernel3=[1, 1, 0, 0], preprocess_type=preprocess_type, **kwargs)
