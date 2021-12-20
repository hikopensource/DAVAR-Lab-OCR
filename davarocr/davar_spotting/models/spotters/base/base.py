"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    base.py
# Abstract       :    base model definition for End to End model

# Current Version:    1.0.0
# Date           :    2021-03-19
##################################################################################################
"""
import logging
from abc import ABCMeta
from mmdet.models.detectors.base import BaseDetector

from mmcv.runner import auto_fp16


class BaseEndToEnd(BaseDetector):
    """Base class for end_to_end recognizors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseEndToEnd, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_transformation(self):
        return hasattr(self, 'recog_transformation') and self.recog_transformation is not None

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)
