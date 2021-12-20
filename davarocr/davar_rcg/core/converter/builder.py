"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from mmcv.utils import Registry, build_from_cfg

CONVERTER = Registry('converter')


def build_converter(cfg):
    """
    Args:
        cfg (config): model config):

    Returns:
        build the converter

    """
    assert 'type' in cfg and isinstance(cfg['type'], str)
    converter = build_from_cfg(cfg, CONVERTER)

    return converter
