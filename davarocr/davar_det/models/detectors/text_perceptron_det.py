"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    text_perceptron_det.py
# Abstract       :    The main pipeline definition of tp_det model

# Current Version:    1.0.0
# Date           :    2020-05-31
####################################################################################################
"""
from mmdet.models.builder import DETECTORS
from .seg_based_det import SegBasedDet


@DETECTORS.register_module()
class TextPerceptronDet(SegBasedDet):
    """Implementation of Text Perceptron detector model.[1]

    Ref: [1] Text Perceptron: Towards End-to-End Arbitrary Shaped Text Spotting. AAAI-20.
                <https://arxiv.org/abs/2002.06820>`_
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """ Implementation of Text Perceptron detector model.

        Args:
            backbone(dict): network backbone (e.g. ResNet)
            neck(dict): network neck (e.g., FPN)
            mask_head(dict): head for loss calculation (e.g., TPHead)
            train_cfg(dict): related parameters for training
            test_cfg(dict): related parameters for test
            pretrained(dict): pretrained model
        """
        super().__init__(backbone=backbone, neck=neck, mask_head=mask_head,
                         train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained)

    def aug_test(self, img, img_meta):
        raise NotImplementedError
