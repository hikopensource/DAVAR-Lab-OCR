"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-03-19
##################################################################################################
"""
from .base import BaseEndToEnd, TwoStageEndToEnd, SegBasedEndToEnd
from .mask_rcnn_spot import MaskRCNNSpot
from .text_perceptron_spot import TextPerceptronSpot
from .mango import MANGO


__all__ = ['BaseEndToEnd', 'SegBasedEndToEnd', 'TwoStageEndToEnd', 'MANGO', 'MaskRCNNSpot', 'TextPerceptronSpot']
