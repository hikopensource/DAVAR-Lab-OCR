# -*- coding: utf-8 -*-
"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    east_head.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-06-08
##################################################################################################
"""

import torch
import torch.nn as nn
from mmdet.models.builder import build_loss,HEADS


@HEADS.register_module()
class EASTHead(nn.Module):
    """ Implementation of EAST head [1]

    Ref: [1] An Efficient and Accurate Scene Text Detector. CVPR-2017
    """
    def __init__(self, loss_seg, loss_reg, geometry='RBOX'):
        """
        Args:
            loss_seg(dict): loss for score map training
            loss_reg(dict): loss for geo map trainning
            geometry(dict): geo map mode, in "RBOX" or "QUAD"
        """
        super().__init__()
        self.loss_seg = build_loss(loss_seg)
        self.loss_reg = build_loss(loss_reg)
        self.geometry = geometry
        assert self.geometry in ['RBOX', 'QUAD'], "Only support geometry mode of 'RBOX' or 'QUAD'"
        self.score_map = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.score_map_sigmoid = nn.Sigmoid()
        if self.geometry == 'RBOX':
            self.geo_map = nn.Conv2d(32, 4, kernel_size=1, stride=1)
            self.angle_map = nn.Conv2d(32, 1, kernel_size=1, stride=1)
            self.angle_map_sigmod = nn.Sigmoid()
            self.geo_map_sigmod = nn.Sigmoid()
        else:
            self.geo_map = nn.Conv2d(32, 8, kernel_size=1, stride=1)
            self.angle_map = None
            self.angle_map_sigmod = None
            self.geo_map_sigmod = None


    def init_weights(self):
        """ Weight initialization """
        for module in self.modules():
            if module is None:
                continue
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def _power_layer(x, power=1, scale=512.0, shift=0.0):
        """ Rescale Tensor
            y = [(scale*x) + shift]^power
        Args:
            x (Tensor): input feature map
            power(int): power number
            scale(float): scale factor
            shift(float) shift factor

        Returns:
            Tensor: Rescaled feature map
        """
        return torch.pow(torch.add(shift, torch.mul(scale, x)), power)

    def forward(self, x):
        """ Forward compute of EAST head

        Args:
            x(Tensor): input feature map in shape of B x 32 x H/4 x W/4

        Returns:
            Tensor: score map in shape of [B, 1, H/4, W/4]
        Returns:
            Tensor: geo map in shape of [B, 5, H/4. W/4] or [B, 8, H/4. H/4]
        """
        score_map = self.score_map_sigmoid(self.score_map(x))
        if self.geometry == 'RBOX':
            geo_map_pre = self.geo_map_sigmod(self.geo_map(x))
            angle_map_pre = self.angle_map_sigmod(self.angle_map(x))

            # Balance distance and angle
            geo_map = torch.cat((self._power_layer(geo_map_pre),
                                self._power_layer(angle_map_pre, 1, 1.570796327, -0.7853981635)),
                                dim=1)
        else:
            geo_map = self.geo_map(x)
        return score_map, geo_map

    def get_target(self, gt_masks):
        """ Get split targets from gt_masks

        Args:
            gt_masks(Tensor): in shape of [B, 14, H/4, W/4] or [B, 20, H/4, W/4], where
                     gt_mask:[:,0]    :  gt_score_map
                     gt_mask:[:,1]    :  gt_score_map_mask, 1 Care / 0 Not Care
                     gt_mask:[:,2:7]  or gt_mask[:, 2:10] :  gt_geo_map
                     gt_mask:[:,7:12]  or gt_mask[:, 10:18] :  gt_geo_map

        Returns:
            Tensor: score map, in shape of [B, 1, H/4, W/4]
        Returns:
            Tensor: score map mask, in shape of [B, H/4, W/4]
        Returns:
            Tensor: geo map, in shape of [B, 5, H/4, W/4] or [B, 8, H/4, W/4]
        Returns:
            Tensor: geo map weights, in shape of [B, 5, H/4, W/4] or [B, 8, H/4, W/4]
        """
        score_map = gt_masks[:, 0:1, :, :]
        score_map_masks = gt_masks[:, 1, :, :].float()
        if self.geometry == 'RBOX':
            geo_map = gt_masks[:, 2:7, :, :]
            geo_map_weights = gt_masks[:, 7:12, :, :].float()
        else:
            geo_map = gt_masks[:, 2:10, :, :]
            geo_map_weights = gt_masks[:, 10: 18, :, :].float()

        return score_map, score_map_masks, geo_map, geo_map_weights

    def loss(self, mask_pred, mask_targets):
        """ Compute loss

        Args:
            mask_pred(Tuple(Tensor)):  score_map in shape of [B, 1, H/4, W/4],
                                       geo_map in shape of [B, 5, H/4. W/4] or [B, 8, H/4. H/4]
            mask_targets(Tuple(Tensor)): score map, in shape of [B, 1, H/4, W/4]
                                         score map mask, in shape of [B, H/4, W/4]
                                         geo map, in shape of [B, 5, H/4, W/4] or [B, 8, H/4, W/4]
                                         geo map weights, in shape of [B, 5, H/4, W/4] or [B, 8, H/4, W/4]

        Returns:
            dict: loss in a dict
        """
        score_pred, reg_pred = mask_pred
        score_map, score_map_masks, geo_map, geo_map_weights = mask_targets
        loss = dict()

        # score_map loss
        loss["loss_seg_text"] = self.loss_seg(score_pred, score_map, weight=score_map_masks)

        # geo_map loss
        loss["loss_reg"] = self.loss_reg(reg_pred, geo_map, weight=geo_map_weights)
        return loss

    def get_seg_masks(self, mask_pred, img_meta):
        """ The main process of box generation is implemented in postprocessing"""
        return mask_pred, img_meta
