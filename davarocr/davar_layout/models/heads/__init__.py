"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-12-06
##################################################################################################
"""
from .convfc_bbox_head_w_gcn import Shared2FCBBoxHeadWGCN, ConvFCBBoxHeadWGCN
from .cascade_roi_head_w_gcn import CascadeRoIHeadWGCN
from .standard_roi_head_w_gcn import StandardRoIHeadWGCN
from .gcn_head import GCNHead

__all__ = ['GCNHead', 'ConvFCBBoxHeadWGCN', 'Shared2FCBBoxHeadWGCN', 'CascadeRoIHeadWGCN', 'StandardRoIHeadWGCN']
