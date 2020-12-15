"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    text_perceptron_det.py
# Abstract       :    the main pipeline definition of tp_det model

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31

# Modified Date  :    2020-11-25
# Modified by    :    inusheng
# Comments       :    Code and comment standardized
####################################################################################################
"""
import os
import time
import importlib

import torch.nn as nn

from mmdet.models.detectors.base import BaseDetector
from mmdet.models import builder, build_roi_extractor
from mmdet.models.registry import DETECTORS


@DETECTORS.register_module
class TextPerceptronDet(BaseDetector):
    """
        Description:
            Text Perceptron Detector model structure

        Properties:
            backbone: network backbone (e.g. ResNet)
            neck: network neck (e.g., FPN)
            mask_head: head for loss calculation (e.g., TPHead)
            train_cfg: related parameters for training
            test_cfg: related parameters for test
            pretrained: pretrained model
            
            Note: def of backbone, neck, ... are terms used in standard mmdetection framework.
                  You are recommended to be familiar with mmdetection by searching for some quick-run tutorials.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 mask_head=None,
                 shape_transform_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.mask_head = builder.build_head(mask_head)
        if shape_transform_module is not None:
            self.shape_transform_module = build_roi_extractor(shape_transform_module)
        else:
            self.shape_transform_module = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """
        Description:
            network parameters initialization
        Args:
            pretrained：pretrained model
        """
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for module in self.neck:
                    module.init_weights()
            else:
                self.neck.init_weights()
        self.mask_head.init_weights()

    def extract_feat(self, img):
        """
        Description:
            feature extraction, mainly including backbone part and neck part
        
        Args:
            img: input images
        
        Returns:
            x: output feature maps through feature extractor
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      **kwargs
                      ):
        """
        Description:
            forward pass and loss computing (this forward function is used for training)
            
        Arguments:
            img (np.ndarray): input images
            img_metas(dict) : image meta-info
            gt_masks(np.ndarray): ground-truth label for training
        
        Returns:
            losses(dict): losses for training data
        """
        losses = dict()
        x = self.extract_feat(img)

        # compute features through mask_head
        mask_pred = self.mask_head(x)

        # get ground-truth label
        mask_targets = self.mask_head.get_target(gt_masks)

        # compute loss
        loss_mask = self.mask_head.loss(mask_pred, mask_targets)

        # update loss
        losses.update(loss_mask)

        
        # For End-to-End training,
        # Text Recognition part is unavailable due to the confidential policy
        # [SUSPENDED] Hopefully this part will be relesed later in the feature.

        # roi_features = self.shape_transform_module(mask_pred)
        # recog_features = self.recog_backbone(roi_features)

        return losses

    def forward_dummy(self, img):
        """
        Description:
            dummy forward pass (mainly for FLOPS calculation)
        """
        x = self.extract_feat(img)
        outs = self.mask_head(x)
        return outs

    def simple_test(self, img, img_meta, **kwargs):
        """
        Description:
            forward inference (for test)
 
        Args:
            img: input image
            img_meta: image meta-info
            
        Returns:
            results: predicted results, { points：[x1, y1, x2, y2, ..., xn, yn]}
        """
        x = self.extract_feat(img)
        mask_pred = self.mask_head(x)

        points = self.shape_transform_module(mask_pred, img_meta)

        # For End-to-End testing
        # Text Recognition part is unavailable due to the confidential policy
        # [SUSPENDED] Hopefully this part will be relesed later in the feature.

        # points, crop_regions = self.shape_transform_module(mask_pred, img_meta)

        return points

    def aug_test(self, img, img_meta):
        raise NotImplementedError