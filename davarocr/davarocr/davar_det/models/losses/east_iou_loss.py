# -*- coding: utf-8 -*-
"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    east_iou_loss.py
# Abstract       :    Iou loss for EAST

# Current Version:    1.0.0
# Date           :    2021-05-18
##################################################################################################
"""
import math
import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class EASTIoULoss(nn.Module):
    """ EAST IOU loss for regression. [1], only support for 'RBOX' mode

    Ref: [1] An Efficient and Accurate Scene Text Detector. CVPR-2017
    """
    def __init__(self, mode='iou', eps=1e-5,  loss_weight=1.0, loss_angle_weight=10.0):
        """

        Args:
            mode(str): IOU mode, support for 'iou', 'giou', 'diou', 'ciou'
            eps(float): factor to prevent division of 0.
            loss_weight(float): IOU loss weight
            loss_angle_weight (float): weight to balance angle weight.
        """
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight
        self.loss_angle_weight = loss_angle_weight
        self.mode = mode
        assert self.mode in ['iou', 'giou', 'diou', 'ciou'], \
            "Only support modes in 'iou','giou', 'diou' and 'ciou'"

    def forward(self, pred, target, weight=None):
        """ Loss computation

        Args:
            pred (Tensor): prediction feature map, in shape of [B, 5, H, W]
            target (Tensor):  target feature map. in shape of [B, 5, H, W]
            weight (Tensor): geo map weights, in shape of [B, 5, H, W]

        Returns:
            Tensor: loss
        """

        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(target, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(pred, 1, 1)

        # Calculate IoU
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        ious = area_intersect/(area_union + self.eps)

        # Other types of Iou
        if self.mode == 'iou':
            # normal iou
            loss_area = -torch.log(ious+self.eps)
        elif self.mode == 'giou':
            # enclose area
            enclose_w = torch.max(d2_gt, d2_pred) + torch.max(d4_gt, d4_pred)
            enclose_h = torch.max(d1_gt, d1_pred) + torch.max(d3_gt, d3_pred)

            enclose_area = enclose_w * enclose_h + self.eps
            gious = ious - (enclose_area - area_union) / enclose_area
            loss_area = 1 - gious
        elif self.mode == 'diou':
            # diou
            enclose_w = torch.max(d2_gt, d2_pred) + torch.max(d4_gt, d4_pred)
            enclose_h = torch.max(d1_gt, d1_pred) + torch.max(d3_gt, d3_pred)

            enclose_d = enclose_w **2 + enclose_h**2
            center_d = ((d4_pred-d2_pred)/2-(d4_gt-d2_gt)/2)**2 + ((d3_pred-d1_pred)/2-(d3_gt-d1_gt)/2)**2
            dious = ious - center_d / (enclose_d + self.eps)
            loss_area = 1 - dious
        else:
            # ciou:
            enclose_w = torch.max(d2_gt, d2_pred) + torch.max(d4_gt, d4_pred)
            enclose_h = torch.max(d1_gt, d1_pred) + torch.max(d3_gt, d3_pred)

            enclose_d = enclose_w **2 + enclose_h**2
            center_d = ((d4_pred-d2_pred)/2-(d4_gt-d2_gt)/2)**2 + ((d3_pred-d1_pred)/2-(d3_gt-d1_gt)/2)**2
            u = center_d / (enclose_d+self.eps)
            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan((d4_gt + d2_gt) / (d1_gt + d3_gt + self.eps)) - torch.atan(
                (d4_pred + d2_pred) / (d1_pred + d3_pred + self.eps))), 2)
            with torch.no_grad():
                S = 1-ious
                alpha = v/(S+v+self.eps)
            cious = ious - (u + alpha * v)
            loss_area = 1 - cious
        L_AABB = loss_area
        L_theta = 1 - torch.cos(theta_pred - theta_gt)
        L_g = L_AABB + self.loss_angle_weight * L_theta

        if weight is not None:
            return self.loss_weight * torch.mean(L_g * weight)

        return self.loss_weight * torch.mean(L_g)
