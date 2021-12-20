"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ResNet32.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-06-01
##################################################################################################
"""
from mmdet.models.builder import BACKBONES
from davarocr.davar_rcg.models.backbones import ResNet32
from davarocr.davar_rcg.models.backbones.ResNet32 import BasicBlock
from davarocr.davar_rcg.models.backbones.ResNet32 import ResNet


@BACKBONES.register_module()
class CustomResNet32(ResNet32):
    """Same with ResNet32, The only difference is that we return more a lower feature for tracking learning and qscore
    learning
    """
    def __init__(self, input_channel, output_channel=512):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
        """
        super().__init__(input_channel=input_channel, output_channel=output_channel)
        self.ConvNet = CustomResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])


class CustomResNet(ResNet):
    """Custom Res-Net network structure which return two level features"""
    def __init__(self, input_channel,
                 output_channel, block, layers):
        """

        Args:
            input_channel (int): input channel
            output_channel (int): output channel
            block (BasicBlock): convolution block
            layers (list): layers of the block
        """
        super().__init__(input_channel=input_channel, output_channel=output_channel, block=block, layers=layers)

    def forward(self, x):
        """
        Args:
            x (tensor): input feature

        Returns:
            Tensor: output feature of the Resnet

        Returns:
            Tensor: output feature for tracking and qscore
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

        low_feature = x

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

        return x, low_feature
