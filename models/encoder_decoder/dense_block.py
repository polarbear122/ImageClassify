import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_planes=3, growth_rate=16):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=(3, 3), padding=1, bias=False)

    def forward(self, x):
        print(x.shape)
        out1 = self.conv1(F.relu(self.bn1(x)))
        print(out1.shape)
        out2 = self.conv2(F.relu(self.bn2(out1)))
        print(out2.shape)
        out3 = torch.cat([out2, x], 1)
        print(out3.shape)
        return out3


def DenseBlock_init():
    return DenseBlock()
