"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from .seg_heads import tp_head, east_head
from .detectors import TextPerceptronDet, EAST, MaskRCNNDet
from .losses import DiceLoss, EASTIoULoss
from .neck import EastMerge

__all__ = ['tp_head', 'east_head', 'TextPerceptronDet', 'EAST', 'MaskRCNNDet', 'DiceLoss','EASTIoULoss',
           'EastMerge']
