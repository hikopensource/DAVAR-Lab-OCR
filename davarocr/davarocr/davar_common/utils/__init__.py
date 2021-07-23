"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from mmcv.utils import Registry, build_from_cfg
from .collect_env import collect_env
from .logger import get_root_logger

__all__ = [
    'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env'
]
