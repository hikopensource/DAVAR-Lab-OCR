"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ResNetRFL.py
# Abstract       :    Implementations of Resnet in RF-Learning

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import logging
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.models.builder import BACKBONES

from .ResNet32 import BasicBlock


@BACKBONES.register_module()
class ResNetRFL(nn.Module):
    """
    Backbone network of the reciprocal feature learning in Ref [1]

    Ref [1]: Reciprocal Feature Learning via Explicit and Implicit Tasks in Scene Text Recognition. ICDAR-2021.
    """

    def __init__(self, input_channel, output_channel=512):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
        """
        super(ResNetRFL, self).__init__()
        self.backbone = RFLBase(input_channel)

        self.out_channel = output_channel
        self.output_channel_block = [int(self.out_channel / 4), int(self.out_channel / 2),
                                     self.out_channel, self.out_channel]
        block = BasicBlock
        layers = [1, 2, 5, 3]
        self.inplanes = int(self.out_channel // 2)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2],
                               self.output_channel_block[2],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3],
                                 self.output_channel_block[3],
                                 kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3],
                                 self.output_channel_block[3],
                                 kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

        self.inplanes = int(self.out_channel // 2)

        self.v_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.v_layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.v_conv3 = nn.Conv2d(self.output_channel_block[2],
                                 self.output_channel_block[2],
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.v_bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.v_layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.v_conv4_1 = nn.Conv2d(self.output_channel_block[3],
                                   self.output_channel_block[3],
                                   kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.v_bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.v_conv4_2 = nn.Conv2d(self.output_channel_block[3],
                                   self.output_channel_block[3],
                                   kernel_size=2, stride=1, padding=0, bias=False)
        self.v_bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

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
            for para in self.modules():
                if isinstance(para, nn.Conv2d):
                    kaiming_init(para, mode="fan_in",
                                 nonlinearity='relu',
                                 distribution='normal')  # leaky_relu
                elif isinstance(para, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(para, 1, bias=0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): input feature

        Returns:
             torch.Tensor: ResNet_RFL output feature

        """

        x_1 = self.backbone(inputs)

        # visual stage 1
        v_x = self.v_maxpool3(x_1)
        v_x = self.v_layer3(v_x)
        v_x = self.v_conv3(v_x)
        v_x = self.v_bn3(v_x)
        visual_feature_2 = self.relu(v_x)
        # x_2 torch.Size([batch, 512, h/8, 26])

        # visual stage 2
        v_x = self.v_layer4(visual_feature_2)
        v_x = self.v_conv4_1(v_x)
        v_x = self.v_bn4_1(v_x)
        v_x = self.relu(v_x)
        v_x = self.v_conv4_2(v_x)
        v_x = self.v_bn4_2(v_x)
        visual_feature_3 = self.relu(v_x)

        # ====== semantic branch =====
        x = self.maxpool3(x_1)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x_2 = self.relu(x)
        # x_2 torch.Size([batch, 512, h/8, 26])

        x = self.layer4(x_2)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x_3 = self.relu(x)

        return [visual_feature_3, x_3]


class ResNetBase(nn.Module):
    """ Base share backbone network"""
    def __init__(self, input_channel,
                 output_channel, block, layers):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
            block (BasicBlock): convolution block
            layers (list): layers of the block
        """
        super(ResNetBase, self).__init__()

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
            torch.Tensor: output feature of the Resnet_base

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

        return x


class RFLBase(nn.Module):
    """ Reciprocal feature learning share backbone network"""
    def __init__(self, input_channel, output_channel=512):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
        """
        super(RFLBase, self).__init__()
        self.ConvNet = ResNetBase(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): input feature

        Returns:
            torch.Tensor: output feature of the RFL_Base

        """
        return self.ConvNet(inputs)
