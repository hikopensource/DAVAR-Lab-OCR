"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from .seg_heads import yoro_recommender_head, spatial_tempo_east_head
from .detectors import spatial_temporal_east_det
from .recognizors import TextRecommender
from .losses import triple_loss
from .backbones import CustomResNet32

__all__=['spatial_temporal_east_det', 'spatial_tempo_east_head', 'yoro_recommender_head', 'TextRecommender',
         'triple_loss']
