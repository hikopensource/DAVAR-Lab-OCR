"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    centerline_seg_head.py
# Abstract       :    Global centerline segmentation

# Current Version:    1.0.0
# Date           :    2021-03-19
######################################################################################################
"""
import torch
import torch.nn as nn
import cv2
import pyclipper
import numpy as np
from mmdet.models.builder import build_loss, HEADS
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16


@HEADS.register_module()
class CenterlineSegHead(nn.Module):
    """ Implement of Word level centerline segmentation, refer to PSENet [1].

    Ref: [1] PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network. CVPR-19
             <https://arxiv.org/abs/1806.02559>`_
    """

    def __init__(self,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=2,
                 sigma=0.1,
                 featmap_indices=(0, 1, 2, 3),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_seg=None):
        """
        Args:
            in_channels (int): input feature map channels
            conv_out_channels (int):  output feature map channels
            num_classes (int): mask category numbers
            sigma (float): shrink parameter
            featmap_indices (tuple(int)): feature map levels indices
            conv_cfg (dict): configuration of convolutions
            norm_cfg (dict): configuration of normalization
            loss_seg (dict): loss of segmentation
        """

        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.featmap_indices = featmap_indices
        self.num_classes = num_classes
        self.out_channels = self.num_classes - 1
        self.sigma = sigma
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if loss_seg is not None:
            self.loss_seg = build_loss(loss_seg)
        else:
            self.loss_seg = None

        # Convolutions using for capture long height-width features
        self.P4_conv = ConvModule(self.in_channels,
                                  self.conv_out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg)
        self.P4_1x7_conv = ConvModule(self.conv_out_channels,
                                      self.conv_out_channels,
                                      kernel_size=(1, 7),
                                      stride=(1, 1),
                                      padding=(0, 3),
                                      conv_cfg=self.conv_cfg,
                                      norm_cfg=self.norm_cfg)
        self.channel4_1x7_conv = ConvModule(self.in_channels,
                                            self.conv_out_channels,
                                            kernel_size=(1, 7),
                                            stride=(1, 1),
                                            padding=(0, 3),
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg)
        self.rpn4 = ConvModule(self.conv_out_channels,
                               self.conv_out_channels,
                               kernel_size=3,
                               padding=1,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg)
        self.conv_logits_text = nn.Conv2d(self.conv_out_channels,
                                          self.out_channels,
                                          kernel_size=1)

    def init_weights(self):
        """ Weight initialization """
        for module in [self.conv_logits_text]:
            if module is not None:
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward_single(self, feat):
        """ Forward computation in single level

        Args:
            feat (Tensor): input feature maps, in shape of [B, C, H, W]

        Returns:
            Tensor: predict feature map, in shape of [B, 1, H, W]
        """

        x_4 = feat
        x_p4 = self.P4_conv(x_4)
        x_4_1x7 = self.channel4_1x7_conv(x_4)
        x_p4_1x7 = self.P4_1x7_conv(x_p4)
        x_4 = x_p4_1x7 + x_p4 + x_4_1x7
        x_4 = self.rpn4(x_4)
        score_pred_text_4 = self.conv_logits_text(x_4)
        score_pred_text_4 = torch.sigmoid(score_pred_text_4)
        return score_pred_text_4

    @auto_fp16()
    def forward(self, feats):
        """ Forward computation in multiple levels

        Args:
            feats (list(Tensor)): input feature maps, in shape of [B, C, H, W]

        Returns:
            list(Tensor): predict feature map, in shape of [B, 2, H, W]
        """

        preds = []
        for i in range(len(self.featmap_indices)):
            pred = self.forward_single(feats[i])
            preds.append(pred)

        return preds

    def _shrink_bboxes(self, gt_poly_bboxes):
        """ Segmentation Label Generate by shrinking poly boxes.

        Args:
            gt_poly_bboxes (list(list(float)): ground-truth of boxes for text instances

        Returns:
            list(list(float)): shrinked boxes for text instances.
        """

        shrinked_bboxes = []
        for batch_bboxes in gt_poly_bboxes:
            tmp_batch_bboxes = []
            for poly_bbox in batch_bboxes:
                poly_bbox = np.array(poly_bbox, dtype=np.int).reshape(-1, 2)
                # Calculate the area and perimeter of polygon
                area = cv2.contourArea(poly_bbox)
                permimeter = cv2.arcLength(poly_bbox, True)
                # Clip polygon according to its area and perimeter
                if permimeter != 0:
                    dis = area * (1 - self.sigma ** 2) / permimeter
                    pco = pyclipper.PyclipperOffset()
                    pco.AddPath(poly_bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                    shrinked_poly = pco.Execute(-dis)
                    tmp_batch_bboxes.append(shrinked_poly)
                else:
                    tmp_batch_bboxes.append([])
            shrinked_bboxes.append(tmp_batch_bboxes)
        return shrinked_bboxes

    def _get_target_single(self, gt_bboxes, feat_size, stride, device='cuda'):
        """ Ground-truth of mask generated according to word level annotations in single level.

        Args:
            gt_bboxes (list(list(float)): A list of polygon bboxes, in shape of [[x1, y1, x2, y2,...., xn, yn], ...]
            feat_size (tuple): feature map shape
            stride (int): An int number the indicate the feature map stride

        Returns:
            Tensor: learning target, in shape of [B, H, W]
        """

        batch, _, height, width = feat_size
        gt_mask = torch.zeros([batch, height, width], dtype=torch.long, device=device)

        shrink_bboxes = self._shrink_bboxes(gt_bboxes)
        for batch_id in range(batch):
            batch_bboxes = shrink_bboxes[batch_id]
            target_mask = np.zeros((height, width), dtype=np.uint8)
            # Fill gt mask according to the cropped polygon
            for bboxes in batch_bboxes:
                if len(bboxes) == 1:
                    bboxes = np.array(bboxes)
                    bboxes_downsample = (bboxes / float(stride)).astype(int)
                    cv2.fillPoly(target_mask, [bboxes_downsample], color=1)
            target_mask = torch.Tensor(target_mask)
            gt_mask[batch_id, ...] = target_mask
        return gt_mask

    def get_target(self, feats, gt_bboxes):
        """ Ground-truth of mask generated according to word level annotations in multiple levels.

        Args:
            feats (list(Tensor)): input feature maps, in shape of [B, C, H, W]
            gt_bboxes (list(list(float)):  A list of polygon bboxes, in shape of [[x1, y1, x2, y2,...., xn, yn], ...]

        Returns:
            list(Tensor):  learning target in multiple levels, in shape of [B, H, W]
        """

        mask_targets = []
        for i, stride_idx in enumerate(self.featmap_indices):
            stride = 4 * (2**stride_idx)
            target = self._get_target_single(
                gt_bboxes,
                feats[i].shape,
                stride,
                device=feats[i].device
            )
            mask_targets.append(target)
        return mask_targets

    def loss(self, mask_preds, mask_targets):
        """ Loss computation

        Args:
            mask_preds (list(Tensor)): feature map predictions, in shape of [B, 1, H, W]
            mask_targets (list(Tensor)): feature map targets, in shape of [B, H, W]

        Returns:
            dict: all losses in a dict
        """

        loss = dict()
        for i, stride_idx in enumerate(self.featmap_indices):
            stride = 4 * (2**stride_idx)
            mask_pred = mask_preds[i]
            mask_target = mask_targets[i]

            # B x 1 x H x W -> B x H x W x 1 -> B x H x W
            mask_pred = mask_pred.permute(0, 2, 3, 1).squeeze(-1)
            loss_seg = self.loss_seg(mask_pred, mask_target)
            loss.update({"loss_seg_{}x".format(stride):loss_seg})
        return loss
