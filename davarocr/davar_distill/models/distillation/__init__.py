"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-07-07
##################################################################################################
"""
from .base import BaseDistillation
from .spot_res_distill import SpotResolutionDistillation

__all__ = ['SpotResolutionDistillation', 'BaseDistillation']
