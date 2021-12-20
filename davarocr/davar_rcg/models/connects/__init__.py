"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
from .stack_block import CascadeRNN, CascadeCNN
from .single_block import BidirectionalLSTM, S2VAdaptor, V2SAdaptor


__all__ = [
    'BidirectionalLSTM',           # single layer
    'V2SAdaptor',
    'S2VAdaptor',

    'CascadeRNN',
    'CascadeCNN',                  # stacked layers
           ]
