"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ResNet32.py
# Abstract       :    Implementations of Resnet-32

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import logging
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class ResNet32(nn.Module):
    """
    Feature Extractor is proposed in  FAN Ref [1]

    Ref [1]: Focusing Attention: Towards Accurate Text Recognition in Neural Images ICCV-2017
    """

    def __init__(self, input_channel, output_channel=512):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
        """
        super(ResNet32, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

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
                                 distribution='normal')  # leaky_relu
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1, bias=0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): input feature

        Returns:
             torch.Tensor: output feature

        """
        return self.ConvNet(inputs)


class BasicBlock(nn.Module):
    """Res-net Basic Block"""
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None,
                 norm_type='BN', **kwargs):
        """
        Args:
            inplanes (int): input channel
            planes (int): channels of the middle feature
            stride (int): stride of the convolution
            downsample (int): type of the down_sample
            norm_type (str): type of the normalization
            **kwargs (None): backup parameter
        """
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        """

        Args:
            in_planes (int): input channel
            out_planes (int): channels of the middle feature
            stride (int): stride of the convolution
        Returns:
            nn.Module: Conv2d with kernel = 3

        """

        return nn.Conv2d(in_planes, out_planes,
                         kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input feature

        Returns:
            torch.Tensor: output feature of the BasicBlock

        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Res-Net network structure"""
    def __init__(self, input_channel,
                 output_channel, block, layers):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
            block (BasicBlock): convolution block
            layers (list): layers of the block
        """
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4),
                                     int(output_channel / 2),
                                     output_channel,
                                     output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block,
                                       self.output_channel_block[0],
                                       layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0],
                               self.output_channel_block[0],
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block,
                                       self.output_channel_block[1],
                                       layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1],
                               self.output_channel_block[1],
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2,
                                     stride=(2, 1),
                                     padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2],
                                       layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2],
                               self.output_channel_block[2],
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3],
                                       layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3],
                                 self.output_channel_block[3],
                                 kernel_size=2, stride=(2, 1),
                                 padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3],
                                 self.output_channel_block[3],
                                 kernel_size=2, stride=1,
                                 padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        """

        Args:
            block (block): convolution block
            planes (int): input channels
            blocks (list): layers of the block
            stride (int): stride of the convolution

        Returns:
            nn.Sequential: the combination of the convolution block

        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input feature

        Returns:
            torch.Tensor: output feature of the Resnet

        """
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        return x
