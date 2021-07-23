"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-03-19
##################################################################################################
"""

from mmdet.models.builder import build
from mmcv.utils import Registry

SPOTTER = Registry('spotter')


def build_spotter(cfg, train_cfg=None, test_cfg=None):
    """
    Build spotter
    Args:
        cfg(dict): config of model
        train_cfg(dict): training configuration
        test_cfg(dict): testing configuration

    Returns:
        obj: spotter

    """

    return build(cfg, SPOTTER, dict(train_cfg=train_cfg, test_cfg=test_cfg))
