"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    light_rcnn.py
# Abstract       :    Light CRNN Model. Only contain 6x conv for text recognation.

# Current Version:    1.0.0
# Date           :    2021-09-01
##################################################################################################
"""
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class LightCRNN(nn.Module):
    """ Lightweight CRNN for text recognation"""

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): input channel
            out_channels (int): output channel
        """
        super().__init__()

        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=(2, 1),
                         padding=(0, 1)),
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=2,
                      stride=(2, 1),
                      padding=(0, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=2,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),)


    def init_weights(self):
        """ Parameters initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode="fan_in",
                             nonlinearity='relu',
                             distribution='normal')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1, bias=0)

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): input feature

        Returns:
            torch.Tensor: output feature of the VGG

        """
        return self.ConvNet(inputs)
