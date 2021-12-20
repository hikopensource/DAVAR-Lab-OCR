"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_head.py
# Abstract       :    Text Perceptron head structure, mainly including losses for segmentation part
                      and regression part

# Current Version:    1.0.0
# Date           :    2020-05-31
######################################################################################################
"""

import numpy as np

import torch
import torch.nn as nn

from mmdet.models.builder import build_loss, HEADS
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32


def make_one_hot(input_tensor, num_classes):
    """Convert a feature map of shape [N, 1, H, W] into its one-hot encoding version of shape [N, C, H, W],
        where C is the number of classes.

    Args:
        input_tensor(Tensor):  input tensor, [N, 1, *]
        num_classes(int) :  the number of classes of feature maps

    Returns:
        Tensor: one-hot encoding of input tensor, [N, num_classes, *]
    """
    input_tensor = input_tensor[:, np.newaxis, ::]
    shape = np.array(input_tensor.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input_tensor.cpu(), 1).to(input_tensor.device)

    return result


@HEADS.register_module()
class TPHead(nn.Module):
    """Text Perceptron detector head structure, This head is used for further feature extraction and generate
       loss according to ground-truth labels. [1]

       Ref: [1] Text Perceptron: Towards End-to-End Arbitrary Shaped Text Spotting. AAAI-20.
                <https://arxiv.org/abs/2002.06820>`_

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
        """ Text Perceptron detector head structure.

        Args:
            in_channels(int)       :  the number of channels of input feature maps
            conv_out_channels(int) :  the number of channels of output feature maps
            conv_cfg(dict)         :  configuration of conv filters
            norm_cfg(dict)         : configuration of normalization
            loss_seg(dict)          :  segmentation loss
            loss_reg_head(dict)     :  regression loss of head area
            loss_reg_tail(dict)     :  regression loss of tail area
            loss_reg_bond(dict)     :  regression loss of center area
        """
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

        # Define extra conv filters for long text feature extraction
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
        """Network parameters initialization
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
        """Network forward pass

        Args:
            x(Tensor): input feature map.

        Return:
            dict: predict featuremaps, including:
                 preds_4x['score_text_pred'], text/non-text classification mask
                 preds_4x['score_head_pred'], head/non-head classification mask
                 preds_4x['score_tail_pred'], tail/non-tail classification mask
                 preds_4x['score_bond_pred'], boundary/non-boundary classification mask
                 preds_4x['reg_head_pred'], regression predictions in head regions.
                 preds_4x['reg_tail_pred'], regression predictions in tail regions.
                 preds_4x['reg_bond_pred'], regression predictions in center-text(offset to the top&bottom boundary) .
        """
        # Compute loss from 4x feature maps only
        # You can add other supervisions on feature maps in terms of your compute resources
        x_4 = x[0]

        # Extract long text feature
        x_p4 = self.P4_conv(x_4)
        x_4_1x7 = self.channel4_1x7_conv(x_4)
        x_p4_1x7 = self.P4_1x7_conv(x_p4)
        x_4 = x_p4_1x7 + x_p4 + x_4_1x7
        x_4 = self.rpn4(x_4)

        # Generate predicted segmentation map
        preds_4x = dict()
        x_4_seg = self.seg_branch_conv(x_4)
        preds_4x['score_text_pred'] = torch.sigmoid(self.conv_logits_text(x_4_seg))  # [N, 1,  H, W]
        preds_4x['score_head_pred'] = torch.sigmoid(self.conv_logits_head(x_4_seg))  # [N, 1, H, W]
        preds_4x['score_tail_pred'] = torch.sigmoid(self.conv_logits_tail(x_4_seg))  # [N, 1, H, W]
        preds_4x['score_bond_pred'] = torch.sigmoid(self.conv_logits_bond(x_4_seg))  # [N, 1, H, W]

        # Generate predicted regression map
        x4_reg = self.seg_branch_conv(x_4)
        preds_4x['reg_head_pred'] = self.conv_regress_head(x4_reg)    # [N, 4, H, W]
        preds_4x['reg_tail_pred'] = self.conv_regress_tail(x4_reg)    # [N, 4, H, W]
        preds_4x['reg_bond_pred'] = self.conv_regress_bond(x4_reg)    # [N, 4, H, W]

        return preds_4x

    def get_target(self, gt_masks):
        """Generate ground-truth labels

        Args:
            gt_masks(Tensor):   input ground-truth labels, where
                                gt_mask:[:,0]    :  gt_score_map
                                gt_mask:[:,1]    :  gt_score_map_mask, 1 Care / 0 Not Care
                                gt_mask:[:,2:6]  :  gt_geo_map_head
                                gt_mask:[:,6:10] :  gt_geo_map_head_weight
                                gt_mask:[:,10:14]:  gt_geo_map_tail
                                gt_mask:[:,14:18]:  gt_geo_map_tail_weight
                                gt_mask:[:,18:22]:  gt_geo_map_bond
                                gt_mask:[:,22:26]:  gt_geo_map_bond_weight
        Returns:
            dict:  all targets in dict, where
                    'score_text_target'      :  one-hot encoding of segmentation map ground-truth of center area of shape [N, 1, H, W]
                    'score_head_target'      :  one-hot encoding of segmentation map ground-truth of head area of shape [N, 1, H, W]
                    'score_tail_target'      :  one-hot encoding of segmentation map ground-truth of tail area of shape [N, 1, H, W]
                    'score_bond_target'      :  one-hot encoding of segmentation map ground-truth of top and bottom boundaries, [N, 1, H, W]
                    'score_map_masks_target' :  mask of segmentation map ground-truth, [N, 1, H, W]
                    'geo_head_target'        :  ground-truth of head corner points regression, [N, 4, H, W]
                    'geo_head_weights_target':  weights of ground-truth of head regression, [N, 4, H, W]
                    'geo_tail_target'        :  gound-truth of tail corner points regression, [N, 4, H, W]
                    'geo_tail_weights_target':  weights of ground-truth of tail regression, [N, 4, H, W]
                    'geo_bond_target'        :  ground-truth of top and bottom boundaries regression, [N, 4, H, W]
                    'geo_bond_weights_target':  weights of ground-truth of top and bottom boundaries regression, [N, 4, H, W]

        """
        total_targets=dict()
        assert len(gt_masks[0]) == 26

        # Segmentation gts
        score_map_target = gt_masks[:, 0, :, :].long()
        score_map_one_hot = make_one_hot(score_map_target, 5).float()
        total_targets['score_text_target'] = score_map_one_hot[:, 1: 2, :, :]
        total_targets['score_head_target'] = score_map_one_hot[:, 2: 3, :, :]
        total_targets['score_tail_target'] = score_map_one_hot[:, 3: 4, :, :]
        total_targets['score_bond_target'] = score_map_one_hot[:, 4: 5, :, :]
        total_targets['score_map_masks_target'] = gt_masks[:, 1, :, :].float()

        # Regression gts
        total_targets['geo_head_target'] = gt_masks[:, 2:6, :, :]
        total_targets['geo_head_weights_target'] = gt_masks[:, 6:10, :, :]
        total_targets['geo_tail_target'] = gt_masks[:, 10:14, :, :]
        total_targets['geo_tail_weights_target'] = gt_masks[:, 14:18, :, :]
        total_targets['geo_bond_target'] = gt_masks[:, 18:22, :, :]
        total_targets['geo_bond_weights_target'] = gt_masks[:, 22:, :, :]

        return total_targets

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_preds, mask_targets):
        """ Loss computation

        Args:
            mask_preds(dict):  All prediction tensors in a dict
            mask_targets(dict):  All targets maps in a dict

        Returns:
            dict: All losses in a dict
        """
        loss = dict()

        # Compute segmentation loss
        loss["loss_seg_text"] = self.loss_seg(mask_preds['score_text_pred'], mask_targets['score_text_target'],
                                              weight=mask_targets['score_map_masks_target'])
        loss["loss_seg_head"] = self.loss_seg(mask_preds['score_head_pred'], mask_targets['score_head_target'],
                                              weight=mask_targets['score_map_masks_target'])
        loss["loss_seg_tail"] = self.loss_seg(mask_preds['score_tail_pred'], mask_targets['score_tail_target'],
                                              weight=mask_targets['score_map_masks_target'])
        loss["loss_seg_bond"] = self.loss_seg(mask_preds['score_bond_pred'], mask_targets['score_bond_target'],
                                              weight=mask_targets['score_map_masks_target'])

        # Compute regression loss
        if self.loss_reg_head is not None:
            loss_reg_head = self.loss_reg_head(mask_preds['reg_head_pred'], mask_targets['geo_head_target'],
                                               weight=mask_targets['geo_head_weights_target'])
            loss["loss_reg_head"] = loss_reg_head
        if self.loss_reg_tail is not None:
            loss_reg_tail = self.loss_reg_tail(mask_preds['reg_tail_pred'], mask_targets['geo_tail_target'],
                                               weight=mask_targets['geo_tail_weights_target'])
            loss["loss_reg_tail"] = loss_reg_tail
        if self.loss_reg_bond is not None:
            loss_reg_bond = self.loss_reg_bond(mask_preds['reg_bond_pred'], mask_targets['geo_bond_target'],
                                               weight=mask_targets['geo_bond_weights_target'])
            loss["loss_reg_bond"] = loss_reg_bond
        return loss

