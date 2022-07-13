"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    resolution_selector.py
# Abstract       :    Definition of resolution selector structure

# Current Version:    1.0.0
# Date           :    2022-07-07
##################################################################################################
"""
import torch.nn as nn

from davarocr.davar_common.models.builder import CONNECTS

@CONNECTS.register_module()
class ResolutionSelector(nn.Module):
    """ Implementation of resolution selector """

    def __init__(self,
                 temperature=5.0,
                 temperature_decay=0.985,
                 scale_factor=[1.0],
                 in_channels=3,
                 out_channels=256):
        """ Resolution selector structure.

        Args:
            temperature (float)         : temperature for gumbel softmax
            temperature_decay (float)   : temperature decay coefficient for gumbel softmax
            scale_factor (list(float))  : image resolution scaling factor
            in_channels (int)           : the number of channels of input feature maps
            out_channels (int)          : the number of channels of output feature maps
        """
        super().__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 8, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(out_channels // 8)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bs1 = BasicBlock(out_channels // 8, out_channels // 8)
        self.bs2 = BasicBlock(out_channels // 8, out_channels // 4)
        self.bs3 = BasicBlock(out_channels // 4, out_channels // 2)
        self.bs4 = BasicBlock(out_channels // 2, out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, len(self.scale_factor))

    def forward(self, feat):
        """Network forward pass

        Args:
            feat (Tensor): input feature map.

        Return:
            Tensor: output feature map.
        """
        feat = self.conv1(feat)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.bs1(feat)
        feat = self.bs2(feat)
        feat = self.bs3(feat)
        feat = self.bs4(feat)
        feat = self.avg_pool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        return feat

class BasicBlock(nn.Module):
    """Res-net Basic Block"""

    def __init__(self,
                 in_channels,
                 out_channels):
        """
        Args:
            in_channels (int): input channel
            out_channels (int): output channel
        """

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self, feat):
        """
        Args:
            feat (torch.Tensor): input feature

        Returns:
            torch.Tensor: output feature of the BasicBlock

        """
        residual = feat

        out = self.conv1(feat)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(feat)
        residual = self.bn3(residual)
        out += residual
        out = self.relu(out)

        return out
