"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :    Add POSTPROCESS module used for different post-process modules

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from mmcv.utils import Registry, build_from_cfg

PREPROCESS = Registry('preprocess')
POSTPROCESS = Registry('postprocess')
CONVERTERS = Registry('converter')

def build_preprocess(cfg):
    """ Build PREPROCESS module

    Args:
       cfg(mmcv.Config): module configuration

    Returns:
       obj: PREPROCESS module
    """
    return build_from_cfg(cfg, PREPROCESS)


def build_postprocess(cfg):
    """ Build POSTPROCESS module

    Args:
       cfg(mmcv.Config): module configuration

    Returns:
       obj: POSTPROCESS module
    """
    return build_from_cfg(cfg, POSTPROCESS)


def build_converter(cfg):
    """
    Args:
        cfg (mmcv.Config): model config):

    Returns:
        obj: CONVERTER module

    """
    return build_from_cfg(cfg, CONVERTERS)
