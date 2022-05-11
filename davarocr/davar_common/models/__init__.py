"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  build_backbone, build_loss)

from mmdet.models.builder import build_detector as build_model

from .builder import CONNECTS, EMBEDDING, TRANSFORMATIONS
from .builder import build_connect, build_embedding, build_transformation

from .loss import StandardCrossEntropyLoss
from .heads import ClsHead

__all__ = ['BACKBONES',
           'DETECTORS',
           'HEADS',
           'LOSSES',
           'NECKS',
           'CONNECTS',
           'EMBEDDING',
           'TRANSFORMATIONS',

           'build_connect',
           'build_backbone',
           'build_loss',
           'build_embedding',
           'build_transformation',

           'StandardCrossEntropyLoss',
           'ClsHead'
           ]
