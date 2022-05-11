"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    affine_transformation.py
# Abstract       :    Implementations of the affine transformation

# Current Version:    1.0.0
# Date           :    2021-03-07
##################################################################################################
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import TRANSFORMATIONS


@TRANSFORMATIONS.register_module()
class Affine_SpatialTransformer(nn.Module):
    """ Rectification Network of RARE, namely Affine-based STN [1]

    Ref: [1] Spatial Transformer Network. NIPS-2016

    Usage:
    transformation=dict(
         type='Affine_SpatialTransformer',
         I_size=(32, 100),
         I_r_size=(32, 100),
         I_channel_num=1),

    """

    def __init__(self,
                 I_size,
                 I_r_size,
                 I_channel_num=1,
                 fix_transformation=False):
        """
        Spatial Transformation Network based on Affine

        Args:
            I_size (tuple): size of input images
            I_r_size (tuple): size of rectified images
            I_channel_num (int): the number of channels of the input image I
            fix_transformation (bool): if fix the parameters of the transformation during training

        """
        super(Affine_SpatialTransformer, self).__init__()
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = AffineLocalizationNetwork(self.I_channel_num, self.I_size)

        # if set STN parameters fixed, the weights and bias of all layers are set to False
        if fix_transformation:
            for p in self.LocalizationNetwork.parameters():
                p.requires_grad = False

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): save path of pretrained model


        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("Affine_SpatialTransformer:")
            load_checkpoint(self, pretrained,
                            strict=False, logger=logger)

    def forward(self, batch_I):
        """

        Args:
            batch_I (tensor): batch of input images[batch_size x I_channel_num x I_r_height x I_r_width]

        Returns:
            torch.Tensor: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]

        """
        xy_batch_C_prime = self.LocalizationNetwork(batch_I).view(-1, 2, 3)

        row_rev = torch.cat([xy_batch_C_prime[:, 1], xy_batch_C_prime[:, 0]], 1).view(-1, 2, 3).permute(0, 2, 1)
        colomn_rev = torch.cat([row_rev[:, 1], row_rev[:, 0]], 1)
        colomn_rev = torch.cat([colomn_rev, row_rev[:, 2]], 1).view(-1, 3, 2).permute(0, 2, 1)
        batch_C_prime = colomn_rev

        build_P_prime = F.affine_grid(batch_C_prime, torch.Size((batch_C_prime.size(0),
                                                                 self.I_channel_num,
                                                                 self.I_r_size[0],
                                                                 self.I_r_size[1])))

        batch_I_r = F.grid_sample(batch_I, build_P_prime, padding_mode='border')

        return batch_I_r


class AffineLocalizationNetwork(nn.Module):
    """
    Localization Network of STN,
    which predicts 6 params of Affine Transformation

    """

    def __init__(self, I_channel_num, I_size):
        """
        Args:
            I_channel_num (int): channel number of input
            I_size (tuple): input image size
        """
        super(AffineLocalizationNetwork, self).__init__()
        self.I_channel_num = I_channel_num
        self.I_size = I_size
        self.loc_conv1_channel = nn.Conv2d(in_channels=self.I_channel_num,
                                           out_channels=48, kernel_size=3,
                                           stride=1, padding=1)
        self.loc_relu1 = nn.ReLU(inplace=True)
        self.loc_pool1 = nn.MaxPool2d(2, 2)
        self.loc_conv2_channel = nn.Conv2d(48, 48,
                                           3, 1, 1)
        self.loc_relu2_new = nn.ReLU(inplace=True)
        self.loc_pool2_new = nn.MaxPool2d(2, 2)
        self.loc_conv3_channel = nn.Conv2d(48, 64,
                                           3, 1, 1)
        self.loc_relu3_new = nn.ReLU(inplace=True)
        self.loc_pool3_new = nn.MaxPool2d(2, 2)
        self.loc_conv4_channel_64 = nn.Conv2d(64, 64,
                                              3, 1, 1)
        self.loc_relu4 = nn.ReLU(inplace=True)
        self.loc_pool4 = nn.MaxPool2d(2, 2)

        self.loc_relu5 = nn.ReLU(inplace=True)
        self.loc_reg = nn.Linear(768, 6)

        # Init fc2 in LocalizationNetwork
        self.loc_reg.weight.data.fill_(0)
        initial_bias = np.array([1, 0, 0, 0, 1, 0])
        self.loc_reg.bias.data = torch.from_numpy(initial_bias).float().view(-1)

        # for item in self.no_update:
        #    item.weight.requires_grad = False
        #    item.bias.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (tensor): input image feature maps [batch_size x I_channel_num x I_height x I_width]

        Returns:
            torch.Tensor: Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = x.size(0)
        x = self.loc_conv1_channel(x)
        x = self.loc_relu1(x)
        x = self.loc_pool1(x)
        x = self.loc_conv2_channel(x)
        x = self.loc_relu2_new(x)
        x = self.loc_pool2_new(x)
        x = self.loc_conv3_channel(x)
        x = self.loc_relu3_new(x)
        x = self.loc_pool3_new(x)
        x = self.loc_conv4_channel_64(x)
        x = self.loc_relu4(x)
        x = self.loc_pool4(x)

        y = x.view(batch_size, -1)
        y = self.loc_relu5(y)
        batch_C_prime = self.loc_reg(y)

        return batch_C_prime
