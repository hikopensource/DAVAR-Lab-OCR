# -*- coding: utf-8 -*-
"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    east_merge.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-06-08
##################################################################################################
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import NECKS


@NECKS.register_module()
class EastMerge(nn.Module):
    """ Simple implementation of EAST FPN

    Args:
        in_channels(list[int]): input feature map channels
    """
    def __init__(self, in_channels):
        super().__init__()

        assert len(in_channels) == 4, 'Only support 4 level feature maps input'

        self.conv_reduce4 = nn.Conv2d(in_channels[3], 256, 1)
        self.conv_reduce3 = nn.Conv2d(in_channels[2], 256, 1)
        self.conv_reduce2 = nn.Conv2d(in_channels[1], 256, 1)
        self.conv_reduce1 = nn.Conv2d(in_channels[0], 256, 1)

        self.conv1 = nn.Conv2d(512, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(320, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

    def init_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        """ Forward computation

        Args:
            feature(tuple(Tensor)): input feature map, (4x, 8x, 16, 32x)

        Returns:
            Tensor: a fused feature map in shape of B x 32 x H/4 x W/4
        """
        # B x 256 x H/4 x W/4, B x 512 x H/8 x W/8,  B x 1024 x H/16 x W/16, B x 2048 x H/32 x W/32,
        c1, c2, c3, c4 = feature

        # B x 256 x H/4 x W/4, B x 256 x H/8 x W/8,  B x 256 x H/16 x W/16, B x 256 x H/32 x W/32,
        c1, c2, c3, c4 = self.conv_reduce1(c1), self.conv_reduce2(c2), self.conv_reduce3(c3), self.conv_reduce4(c4)

        y = F.interpolate(c4, size=c3.shape[2:], mode='bilinear', align_corners=True) # B x 256 x H/16 x H/16
        y = torch.cat((y, c3), 1)  # B x 512 x H/16 x W/16
        y = self.relu1(self.bn1(self.conv1(y)))		# B x 128 x H/16 x W/16
        y = self.relu2(self.bn2(self.conv2(y)))     # B x 128 x H/16 x W/16

        y = F.interpolate(y, size=c2.shape[2:], mode='bilinear', align_corners=True)  # B x 128 x H/8 x W/8
        y = torch.cat((y, c2), 1)  # B x 384 x H/8 x W/8
        y = self.relu3(self.bn3(self.conv3(y)))		# B x 64 x H/8 x W/8
        y = self.relu4(self.bn4(self.conv4(y)))     # B x 64 x H/8 x W/8

        y = F.interpolate(y, size=c1.shape[2:], mode='bilinear', align_corners=True)  # B x 64 x H/4 x W/4
        y = torch.cat((y, c1), 1)  # B x 320 x H/4 x W/4
        y = self.relu5(self.bn5(self.conv5(y)))		 # B x 32 x H/4 x W/4
        y = self.relu6(self.bn6(self.conv6(y)))      # B x 32 x H/4 x W/4

        y = self.relu7(self.bn7(self.conv7(y)))      # B x 32 x H/4 x W/4

        return y
