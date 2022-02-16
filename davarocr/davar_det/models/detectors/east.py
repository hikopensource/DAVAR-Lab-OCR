"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    east.py
# Abstract       :    The main pipeline definition of EAST model

# Current Version:    1.0.0
# Date           :    2020-06-08
####################################################################################################
"""

from mmdet.models.builder import DETECTORS
from .seg_based_det import SegBasedDet


@DETECTORS.register_module()
class EAST(SegBasedDet):
    """ Implementation of EAST [1]

        Ref: [1] An Efficient and Accurate Scene Text Detector. CVPR-2017

    """
    def __init__(self,
                 backbone,
                 neck=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """
        Args:
            backbone(dict): network backbone (e.g. ResNet)
            neck(dict): network neck (e.g., EASTMerge)
            head(dict): head for loss calculation (e.g., EASTHead)
            train_cfg(dict): related parameters for training
            test_cfg(dict): related parameters for test
            pretrained(dict): pretrained model
        """
        super().__init__(backbone=backbone, neck=neck, mask_head=mask_head, train_cfg=train_cfg,
                                   test_cfg=test_cfg, pretrained=pretrained)
