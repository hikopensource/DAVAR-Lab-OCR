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

LOADERS = Registry('loader')
PARSERS = Registry('parser')


def build_loader(cfg):
    """Build anno file loader."""
    return build_from_cfg(cfg, LOADERS)


def build_parser(cfg):
    """Build anno file parser."""
    return build_from_cfg(cfg, PARSERS)
