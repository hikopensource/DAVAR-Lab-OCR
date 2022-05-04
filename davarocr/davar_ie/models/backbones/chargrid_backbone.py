"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    chargrid_backbone.py
# Abstract       :    Chargrid-net encoder implementation, as backbone in extractor

# Current Version:    1.0.0
# Date           :    2022-03-23
##################################################################################################
"""

import torch.nn as nn

from mmdet.models.builder import BACKBONES
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger


@BACKBONES.register_module()
class ChargridEncoder(nn.Module):
    def __init__(self, input_channels, out_indices, base_channels=64):
        super(ChargridEncoder, self).__init__()
        self.C = base_channels
        self.blocks = nn.ModuleList()
        self.out_indices = out_indices

        # additional conv layer for more stride levels
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, self.C, 
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )

        first_a = nn.Sequential(
            nn.Conv2d(self.C, self.C,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.blocks.append(first_a)

        second_a = nn.Sequential(
            nn.Conv2d(self.C, self.C * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.blocks.append(second_a)

        a_dash = nn.Sequential(
            nn.Conv2d(self.C * 2, self.C * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4,
                      kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4,
                      kernel_size=3, stride=1, dilation=2, padding=2),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.blocks.append(a_dash)

        first_a_double_dash = nn.Sequential(
            nn.Conv2d(self.C * 4, self.C * 8,
                      kernel_size=3, stride=2, dilation=4, padding=4),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 8, self.C * 8,
                      kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 8, self.C * 8,
                      kernel_size=3, stride=1, dilation=4, padding=4),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.blocks.append(first_a_double_dash)

        second_a_double_dash = nn.Sequential(
            nn.Conv2d(self.C * 8, self.C * 8,
                      kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 8, self.C * 8,
                      kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 8, self.C * 8,
                      kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.blocks.append(second_a_double_dash)

    def init_weights(self, pretrained=None):
        """
        Weight initialization
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        x = self.conv1(x)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x = block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
