"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

from .structures import BitmapMasksTable
from .lp_mask_target import get_lpmasks

__all__ = ['BitmapMasksTable', 'get_lpmasks']
