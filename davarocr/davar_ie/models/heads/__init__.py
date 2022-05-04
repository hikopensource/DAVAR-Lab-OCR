"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from .ner_head import NERHead
from .chargrid_reg import ChargridRegHead

__all__ = ['NERHead', 'ChargridRegHead']
