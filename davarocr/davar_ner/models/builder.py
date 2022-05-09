"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
from mmcv.utils import Registry, build_from_cfg
from mmdet.models.builder import build

NERS = Registry('ner')


def build_ner(cfg, train_cfg=None, test_cfg=None):
    """
    Args:
        cfg (config): model config
        train_cfg (dict): model training config
        test_cfg (dict): model test config

    Returns:
        build recognition model

    """
    return build(cfg, NERS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
