"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :   Some custommized implementations that different from official mmcv.

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""

from .runner import DavarCheckpointHook, ReduceonplateauLrUpdaterHook

__all__ = [
    'DavarCheckpointHook',
    'ReduceonplateauLrUpdaterHook'
]
