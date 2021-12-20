"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :    Add POSTPROCESS module used for different post-process modules

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from mmcv.utils import Registry
from mmdet.models.builder import build

POSTPROCESS = Registry('postprocess')


def build_postprocess(cfg):
    """ Build POSTPROCESS module

    Args:
       cfg(dict): module configuration

    Returns:
       obj: POSTPROCESS module
    """
    return build(cfg, POSTPROCESS)
