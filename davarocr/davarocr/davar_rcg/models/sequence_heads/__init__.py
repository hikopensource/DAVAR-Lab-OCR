"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
from .att_head import AttentionHead
from .ctc_head import CTCHead
from .warpctc_head import WarpCTCHead
from .counting_head import CNTHead
from .ace_head import ACEHead


__all__ = [
    'AttentionHead',
    'CTCHead',
    'WarpCTCHead',
    'CNTHead',
    'ACEHead',

]
