"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-03-19
##################################################################################################
"""
from .base import BaseEndToEnd
from .two_stage_e2e import TwoStageEndToEnd
from .seg_based_e2e import SegBasedEndToEnd


__all__ = ['BaseEndToEnd', 'SegBasedEndToEnd', 'TwoStageEndToEnd']
