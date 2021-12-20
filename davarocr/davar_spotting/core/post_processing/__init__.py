"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-05-31
##################################################################################################
"""
from .post_mango import PostMango
from .post_mask_rcnn_spot import PostMaskRCNNSpot
from .post_spotter_base import BasePostSpotter

__all__ = ['PostMango', 'PostMaskRCNNSpot', 'BasePostSpotter']
