"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_head.py
# Abstract       :    Text Perceptron head structure, mainly including losses for segmentation part and regression part

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31

# Modified Date  :    2020-11-26
# Modified by    :    inusheng
# Comments       :    Code and comment standardized
######################################################################################################
"""

import numpy as np

import torch
import torch.nn as nn

from mmdet.models.builder import build_loss
from mmdet.models.registry import HEADS
from mmdet.ops import ConvModule
from mmdet.core import force_fp32, auto_fp16


def make_one_hot(input_tensor, num_classes):
    """
    Description:
        convert a feature map of shape [N, 1, H, W] into its one-hot encoding version of shape [N, C, H, W],
        where C is the number of classes.

    Arguments:
        input_tensor:  input tensor, [N, 1, *]
        num_classes :  the number of classes of feature maps
        
    Returns:
        one-hot encoding of input tensor, [N, num_classes, *]
    """
    input_tensor = input_tensor[:, np.newaxis, ::]
    shape = np.array(input_tensor.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input_tensor.cpu(), 1).to(input_tensor.device)

    return result

@HEADS.register_module
class TPHead(nn.Module):
    """
    Description:
        Text Perceptron head structure,
        this head is used for further feature extraction and generate loss wrt ground-truth labels. 
        
    Arguments:
        in_channels      :  the number of channels of input feature maps 
        conv_out_channels:  the number of channels of output feature maps 
        conv_cfg         :  configuration of conv filters
        norm_cfg         ： configuration of normalization 
        loss_seg         :  segmentation loss 
        loss_reg_head    :  regression loss of head area
        loss_reg_tail    :  regression loss of tail area
        loss_reg_bond    :  regression loss of center area
    """
    def __init__(self,
                 in_channels=256,
                 conv_out_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_seg=None,
                 loss_reg_head=None,
                 loss_reg_bond=None,
                 loss_reg_tail=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        assert loss_seg is not None
        self.loss_seg = build_loss(loss_seg)
        self.loss_reg_head = loss_reg_head
        self.loss_reg_bond = loss_reg_bond
        self.loss_reg_tail = loss_reg_tail
        if loss_reg_head is not None:
            self.loss_reg_head = build_loss(loss_reg_head)
        if loss_reg_tail is not None:
            self.loss_reg_tail = build_loss(loss_reg_tail)
        if loss_reg_bond is not None:
            self.loss_reg_bond = build_loss(loss_reg_bond)

        # define extra conv filters for long text feature extraction
        self.P4_conv = ConvModule(self.in_channels, self.conv_out_channels,
                                  kernel_size=3, stride=1, padding=1,
                                  conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg)
        self.P4_1x7_conv = ConvModule(self.conv_out_channels,
                                      self.conv_out_channels,
                                      kernel_size=(1, 7), stride=(1, 1),
                                      padding=(0, 3), conv_cfg=self.conv_cfg,
                                      norm_cfg=self.norm_cfg)
        self.channel4_1x7_conv = ConvModule(self.in_channels,
                                            self.conv_out_channels,
                                            kernel_size=(1, 7), stride=(1, 1),
                                            padding=(0, 3),
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg)
        self.rpn4 = ConvModule(self.conv_out_channels, self.conv_out_channels,
                               3, padding=1, conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg)

        self.seg_branch_conv = ConvModule(self.conv_out_channels,
                                          self.conv_out_channels, 3, padding=1,
                                          conv_cfg=self.conv_cfg,
                                          norm_cfg=self.norm_cfg)
        self.reg_branch_conv = ConvModule(self.conv_out_channels,
                                          self.conv_out_channels, 3, padding=1,
                                          conv_cfg=self.conv_cfg,
                                          norm_cfg=self.norm_cfg)

        self.conv_logits_text = nn.Conv2d(self.conv_out_channels, 1, 1)
        self.conv_logits_head = nn.Conv2d(self.conv_out_channels, 1, 1)
        self.conv_logits_tail = nn.Conv2d(self.conv_out_channels, 1, 1)
        self.conv_logits_bond = nn.Conv2d(self.conv_out_channels, 1, 1)
        self.conv_regress_head = nn.Conv2d(self.conv_out_channels, 4, 1)
        self.conv_regress_tail = nn.Conv2d(self.conv_out_channels, 4, 1)
        self.conv_regress_bond = nn.Conv2d(self.conv_out_channels, 4, 1)

        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """
        Description:
            network parameters initialization
        """
        for module in [self.conv_logits_text, self.conv_logits_head,
                       self.conv_logits_tail, self.conv_logits_bond,
                       self.conv_regress_bond,self.conv_regress_tail,
                       self.conv_regress_head]:
            if module is None:
                continue
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    @auto_fp16()
    def forward(self, x):
        """
        Description:
            network forward pass
        """
        # compute loss from 4x feature maps only
        # you can add other supervisions on feature maps in terms of your compute resources
        x_4 = x[0]

        # extract long text feature
        x_p4 = self.P4_conv(x_4)
        x_4_1x7 = self.channel4_1x7_conv(x_4)
        x_p4_1x7 = self.P4_1x7_conv(x_p4)
        x_4 = x_p4_1x7 + x_p4 + x_4_1x7
        x_4 = self.rpn4(x_4)

        # generate predicted segmentation map
        x_4_seg = self.seg_branch_conv(x_4)
        score_text_pred = self.conv_logits_text(x_4_seg)  # segmentation map for center area [N, 1,  H, W]
        score_head_pred = self.conv_logits_head(x_4_seg)  # segmentation map for head area [N, 1, H, W]
        score_tail_pred = self.conv_logits_tail(x_4_seg)  # segmentation map for tail area [N, 1, H, W]
        score_bond_pred = self.conv_logits_bond(x_4_seg)  # segmentation map for top and bottom boundaries area [N, 1, H, W]

        # generate predicted regression map
        x4_reg = self.seg_branch_conv(x_4)
        reg_head_pred = self.conv_regress_head(x4_reg)    # predicted regression map for head corner points [N, 4, H, W]
        reg_tail_pred = self.conv_regress_tail(x4_reg)    # predicted regression map for tail corner points [N, 4, H, W]
        reg_bond_pred = self.conv_regress_bond(x4_reg)    # predicted regression map for center area [N, 4, H, W]

        return score_text_pred, score_head_pred, score_tail_pred, score_bond_pred, reg_head_pred, reg_tail_pred, reg_bond_pred

    def get_target(self, gt_masks):
        """
        Description:
            generate ground-truth labels
 
        Arguments:
            gt_masks         :  input ground-truth labels 
            gt_mask:[:,0]    :  gt_score_map
            gt_mask:[:,1]    :  gt_score_map_mask, 1 Care / 0 Not Care
            gt_mask:[:,2:6]  :  gt_geo_map_head
            gt_mask:[:,6:10] :  gt_geo_map_head_weight
            gt_mask:[:,10:14]:  gt_geo_map_tail
            gt_mask:[:,14:18]:  gt_geo_map_tail_weight
            gt_mask:[:,18:22]:  gt_geo_map_bond
            gt_mask:[:,22:26]:  gt_geo_map_bond_weight
        Returns:
            score_text_target      :  one-hot encoding of segmentation map ground-truth of center area of shape [N, 1, H, W]
            score_head_target      :  one-hot encoding of segmentation map ground-truth of head area of shape [N, 1, H, W]
            score_tail_target      :  one-hot encoding of segmentation map ground-truth of tail area of shape [N, 1, H, W]
            score_bond_target      :  one-hot encoding of segmentation map ground-truth of top and bottom boundaries, [N, 1, H, W]
            score_map_masks_target :  mask of segmentation map ground-truth, [N, 1, H, W]
            geo_head_target        :  ground-truth of head corner points regression, [N, 4, H, W]
            geo_head_weights_target:  weights of ground-truth of head regression, [N, 4, H, W]
            geo_tail_target        :  gound-truth of tail corner points regression, [N, 4, H, W]
            geo_tail_weights_target:  weights of ground-truth of tail regression, [N, 4, H, W]
            geo_bond_target        :  ground-truth of top and bottom boundaries regression, [N, 4, H, W]
            geo_bond_weights_target:  weights of ground-truth of top and bottom boundaries regression, [N, 4, H, W]

        """
        assert len(gt_masks[0]) == 26
        score_map_target = gt_masks[:, 0, :, :].long()
        score_map_masks_target = gt_masks[:, 1, :, :].float()
        geo_head_target = gt_masks[:, 2:6, :, :]
        geo_head_weights_target = gt_masks[:, 6:10, :, :]
        geo_tail_target = gt_masks[:, 10:14, :, :]
        geo_tail_weights_target = gt_masks[:, 14:18, :, :]
        geo_bond_target = gt_masks[:, 18:22, :, :]
        geo_bond_weights_target = gt_masks[:, 22:, :, :]

        # convert into one-hot encodings 
        score_map_one_hot = make_one_hot(score_map_target, 5).float()
        score_text_target = score_map_one_hot[:, 1: 2, :, :]
        score_head_target = score_map_one_hot[:, 2: 3, :, :]
        score_tail_target = score_map_one_hot[:, 3: 4, :, :]
        score_bond_target = score_map_one_hot[:, 4: 5, :, :]

        return score_text_target, score_head_target, score_tail_target, score_bond_target, score_map_masks_target,\
               geo_head_target, geo_head_weights_target, geo_tail_target, geo_tail_weights_target, geo_bond_target,\
               geo_bond_weights_target


    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets):

        score_text_pred, score_head_pred, score_tail_pred, score_bond_pred, reg_head_pred, reg_tail_pred, reg_bond_pred = mask_pred

        score_text_target, score_head_target, score_tail_target, score_bond_target, score_map_masks_target, \
        geo_head_target, geo_head_weights_target, geo_tail_target, geo_tail_weights_target, geo_bond_target, \
        geo_bond_weights_target = mask_targets

        loss = dict()
        # compute segmentation loss
        loss["loss_seg_text"] = self.loss_seg(score_text_pred, score_text_target, weight=score_map_masks_target)
        loss["loss_seg_head"] = self.loss_seg(score_head_pred, score_head_target, weight=score_map_masks_target)
        loss["loss_seg_tail"] = self.loss_seg(score_tail_pred, score_tail_target, weight=score_map_masks_target)
        loss["loss_seg_bond"] = self.loss_seg(score_bond_pred, score_bond_target, weight=score_map_masks_target)

        # compute regression loss
        if self.loss_reg_head is not None:
            loss_reg_head = self.loss_reg_head(reg_head_pred, geo_head_target,
                                               weight=geo_head_weights_target)
            loss["loss_reg_head"] = loss_reg_head
        if self.loss_reg_tail is not None:
            loss_reg_tail = self.loss_reg_tail(reg_tail_pred, geo_tail_target,
                                               weight=geo_tail_weights_target)
            loss["loss_reg_tail"] = loss_reg_tail
        if self.loss_reg_bond is not None:
            loss_reg_bond = self.loss_reg_bond(reg_bond_pred, geo_bond_target,
                                               weight=geo_bond_weights_target)
            loss["loss_reg_bond"] = loss_reg_bond
        return loss

