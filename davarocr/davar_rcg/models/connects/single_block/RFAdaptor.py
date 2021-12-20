"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    RFAdaptor.py
# Abstract       :    Implementations of RF-Learning reciprocal adaptor

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import logging
import torch.nn as nn


from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import CONNECTS


@CONNECTS.register_module()
class S2VAdaptor(nn.Module):
    """ Semantic to Visual adaptation module"""
    def __init__(self, in_channels=512):
        """RF-Learning s2v adaptor
        Args:
            in_channels (int): input channels
        """
        super(S2VAdaptor, self).__init__()

        self.in_channels = in_channels  # 512

        # feature strengthen module, channel attention
        self.channel_inter = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.channel_bn = nn.BatchNorm1d(self.in_channels)
        self.channel_act = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
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

    def forward(self, semantic):
        """
        Args:
            semantic (Torch.Tensor): recognition feature
        Returns:
            Torch.Tensor: strengthened recognition feature
        """
        semantic_source = semantic  # batch, channel, height, width

        # feature transformation
        semantic = semantic.squeeze(2).permute(0, 2, 1)  # batch, width, channel
        channel_att = self.channel_inter(semantic)       # batch, width, channel
        channel_att = channel_att.permute(0, 2, 1)       # batch, channel, width
        channel_bn = self.channel_bn(channel_att)        # batch, channel, width
        channel_att = self.channel_act(channel_bn)       # batch, channel, width

        # Feature enhancement
        channel_output = semantic_source * channel_att.unsqueeze(-2)  # batch, channel, 1, width

        return channel_output


@CONNECTS.register_module()
class V2SAdaptor(nn.Module):
    """ Visual to Semantic adaptation module"""
    def __init__(self, in_channels=512, return_mask=False):
        """
        RF-Learning v2s adaptor
        Args:
            in_channels (Tensor): input channels
            return_mask (bool): whether to return attention mask
        """
        super(V2SAdaptor, self).__init__()

        # parameter initialization
        self.in_channels = in_channels
        self.return_mask = return_mask

        # output transformation
        self.channel_inter = nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.channel_bn = nn.BatchNorm1d(self.in_channels)
        self.channel_act = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        """
        Args:
            pretrained (str): model path of the pre_trained model
        Returns:
        """

    def forward(self, visual):
        """

        Args:
            visual (Torch.Tensor): visual counting feature
        Returns:
            Torch.Tensor: strengthened visual counting feature
        """

        # Feature enhancement
        visual = visual.squeeze(2).permute(0, 2, 1)  # batch, width, channel
        channel_att = self.channel_inter(visual)     # batch, width, channel
        channel_att = channel_att.permute(0, 2, 1)   # batch, channel, width
        channel_bn = self.channel_bn(channel_att)    # batch, channel, width
        channel_att = self.channel_act(channel_bn)   # batch, channel, width

        # size alignment
        channel_output = channel_att.unsqueeze(-2)   # batch, width, channel

        if self.return_mask:
            return channel_output, channel_att
        return channel_output
