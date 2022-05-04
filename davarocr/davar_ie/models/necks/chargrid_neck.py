"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    chargrid_neck.py
# Abstract       :    Chargrid-net decoder implementation, as neck in extractor

# Current Version:    1.0.0
# Date           :    2022-04-08
##################################################################################################
"""

import torch.nn as nn
import torch
from mmdet.models.builder import NECKS


@NECKS.register_module()
class ChargridDecoder(nn.Module):
    def __init__(self, base_channels=64):
        super(ChargridDecoder, self).__init__()
        self.C = base_channels
        self._init_layers()

    def _init_layers(self):
        # Transposed conv layers in each scale level
        self.s32deconv = nn.Sequential(
            nn.ConvTranspose2d(self.C * 8, self.C * 8, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C * 8),
            nn.ReLU(inplace=True)
        )
        self.s16deconv = nn.Sequential(
            nn.ConvTranspose2d(self.C * 4, self.C * 4, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True)
        )
        self.s8deconv = nn.Sequential(
            nn.ConvTranspose2d(self.C * 2, self.C * 2, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True)
        )
        # Concat feature operations
        self.b1 = nn.Sequential(
            nn.Conv2d(self.C * 12, self.C * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 4, self.C * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.b2 = nn.Sequential(
            # concat features from second_a (4C) and first_b (2C)
            nn.Conv2d(self.C * 6, self.C * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C * 2, self.C * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.c = nn.Sequential(
            # concat features from second_b (2C) and first_a (C)
            nn.Conv2d(self.C * 3, self.C, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
        )
        # Output convs in each scale level
        self.outconv1 = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )
        self.outconv2 = nn.Sequential(
            nn.Conv2d(self.C * 2, self.C, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )
        self.outconv3 = nn.Sequential(
            nn.Conv2d(self.C * 4, self.C, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )
        self.outconv4 = nn.Sequential(
            nn.Conv2d(self.C * 8, self.C, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )
        self.outconv5 = nn.Sequential(
            nn.Conv2d(self.C * 8, self.C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )

    def init_weights(self):
        """
        Weight initialization
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        perform multi-scale feature pyramid operations

        Args:
            x: A tensor of shape [4, ] from upstream encoder
        Returns:
            outs: tuple(torch.Tensor), output features for downstream model operations
        """
        x1, x2, x3, x4 = x

        feat_s16 = self.s32deconv(x4)
        feat_s16 = self.b1(torch.cat([feat_s16, x3], dim=1))

        feat_s8 = self.s16deconv(feat_s16)
        feat_s8 = self.b2(torch.cat([feat_s8, x2], dim=1))

        feat_s4 = self.s8deconv(feat_s8)
        feat_s4 = self.c(torch.cat([feat_s4, x1], dim=1))

        out1 = self.outconv1(feat_s4)
        out2 = self.outconv2(feat_s8)
        out3 = self.outconv3(feat_s16)
        out4 = self.outconv4(x4)
        out5 = self.outconv5(x4)

        outs = (out1, out2, out3, out4, out5)
        return outs