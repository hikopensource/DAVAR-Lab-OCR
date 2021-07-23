"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    seg_based_det.py
# Abstract       :    The main structure definition of segmentation based detector

# Current Version:    1.0.0
# Date           :    2020-05-31
####################################################################################################
"""
import torch.nn as nn

from mmdet.models.detectors.base import BaseDetector
from mmdet.models import builder
from mmdet.models.builder import DETECTORS
from davarocr.davar_common.core import build_postprocess


@DETECTORS.register_module()
class SegBasedDet(BaseDetector):
    """Segmentation-based detector model structure"""

    def __init__(self,
                 backbone,
                 neck=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """ Network Initialization.

        Args:
            backbone(dict): network backbone (e.g. ResNet)
            neck(dict): network neck (e.g., FPN)
            mask_head(dict): head for loss calculation (e.g., TPHead)
            train_cfg(dict): related parameters for training
            test_cfg(dict): related parameters for test
            pretrained(dict): pretrained model
        """
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        if hasattr(self.test_cfg, 'postprocess'):
            self.post_processor = build_postprocess(self.test_cfg.postprocess)
        else:
            self.post_processor = None

    def init_weights(self, pretrained=None):
        """Parameters initialization

        Args:
            pretrained(dict): pretrained model
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
        """Feature extraction, mainly including backbone part and neck part

        Args:
            img(Tensor): input image

        Returns:
            Tensor: output feature maps through feature extractor
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
        """Forward training process and loss computing

        Args:
            img (list[Tensor]): input images
            img_metas(dict) : image meta-info
            gt_masks(np.ndarray): ground-truth label for training

        Returns:
            dict: losses for training data
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

        # For End-to-End training, To be implemented.
        # This implementation is unavailable currently due to the confidential policy

        # roi_features = self.shape_transform_module(mask_pred)
        # recog_features = self.recog_backbone(roi_features)

        return losses

    def forward_dummy(self, img):
        """Dummy forward pass (mainly for FLOPS calculation)

        Args:
            img (Tensor): input image.

        Returns:
            obj: feature map output
        """
        x = self.extract_feat(img)
        outs = self.mask_head(x)
        return outs

    def simple_test(self, img, img_meta, **kwargs):
        """Forward inference

        Args:
            img(Tensor): input image
            img_meta(dict): image meta-info

        Returns:
            dict: predicted results.  e.g. [{'points':[[x1, y1, ...., xn, yn],[]...]}, {},....].
        """
        x = self.extract_feat(img)
        results = self.mask_head(x)

        if self.post_processor is not None:
            results = self.post_processor.post_processing(results, img_meta)

        return results

    def aug_test(self, img, img_meta):
        raise NotImplementedError
