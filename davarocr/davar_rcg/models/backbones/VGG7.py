"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    VGG7.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-05-01
# Thanks to      :    We borrow the released code from http://gitbug.com/clovaai/deep-text-recognition-benchmark
                      for the VGG7.
##################################################################################################
"""
import logging
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class VGG7(nn.Module):
    """
    Feature Extractor proposed in Ref [1]

    Ref [1]: What Is Wrong With Scene Text Recognition Model Comparisons Dataset and Model Analysis ICCV-2019
    """
    def __init__(self, input_channel, output_channel=512):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
        """
        super(VGG7, self).__init__()
        self.output_channel = [int(output_channel / 8),
                               int(output_channel / 4),
                               int(output_channel / 2),
                               output_channel]  # [64, 128, 256, 512]

        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel,
                      self.output_channel[0],
                      3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                 # 64x16x50
            nn.Conv2d(self.output_channel[0],
                      self.output_channel[1],
                      3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                 # 128x8x25
            nn.Conv2d(self.output_channel[1],
                      self.output_channel[2],
                      3, 1, 1),
            nn.ReLU(True),                      # 256x8x25
            nn.Conv2d(self.output_channel[2],
                      self.output_channel[2],
                      3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),       # 256x4x25
            nn.Conv2d(self.output_channel[2],
                      self.output_channel[3],
                      3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),                       # 512x4x25
            nn.Conv2d(self.output_channel[3],
                      self.output_channel[3],
                      3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),        # 512x2x25
            nn.Conv2d(self.output_channel[3],
                      self.output_channel[3],
                      2, 1, 0),
            nn.ReLU(True))                       # 512x1x24

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained,
                            strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode="fan_in",
                                 nonlinearity='relu',
                                 distribution='normal')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): input feature

        Returns:
             torch.Tensor: output feature of the VGG

        """
        return self.ConvNet(inputs)
