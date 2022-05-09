"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-08-02
##################################################################################################
"""
from mmcv.utils import Registry, build_from_cfg
from mmdet.models.builder import build
ENCODERS = Registry('encoder')
DECODERS = Registry('decoder')
PREPROCESSOR = Registry('preprocessor')

def build_encoder(cfg):
    """Build encoder for scene text recognizer."""
    return build_from_cfg(cfg, ENCODERS)


def build_decoder(cfg):
    """Build decoder for scene text recognizer."""
    return build_from_cfg(cfg, DECODERS)


def build_preprocessor(cfg):
    """Build preprocessor for scene text recognizer."""
    return build_from_cfg(cfg, PREPROCESSOR)

