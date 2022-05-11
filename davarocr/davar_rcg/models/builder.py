"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
from mmdet.models.builder import build
from mmcv.utils import Registry


RECOGNIZORS = Registry('recognizor')

def build_recognizor(cfg, train_cfg=None, test_cfg=None):
    """
    Args:
        cfg (config): model config
        train_cfg (dict): model training config
        test_cfg (dict): model test config

    Returns:
        build recognition model

    """
    return build(cfg, RECOGNIZORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
