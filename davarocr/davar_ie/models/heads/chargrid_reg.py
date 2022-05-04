"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    chargrid_reg.py
# Abstract       :    Implementation of bbox regression part in chargrid-net

# Current Version:    1.0.0
# Date           :    2022-04-11
##################################################################################################
"""

from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead


@HEADS.register_module()
class ChargridRegHead(ConvFCBBoxHead):
    def __init__(self, fc_out_channels=256, *args, **kwargs):
        super(ChargridRegHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=3,
            num_cls_fcs=0,
            num_reg_convs=3,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
