"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from .dice_loss import DiceLoss
from .east_iou_loss import EASTIoULoss
__all__ = ['DiceLoss', 'EASTIoULoss']
