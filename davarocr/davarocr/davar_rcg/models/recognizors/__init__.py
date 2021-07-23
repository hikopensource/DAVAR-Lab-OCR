"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
from .general import GeneralRecognizor
from .rf_learning import RFLRecognizor

__all__ = [
    'GeneralRecognizor',
    'RFLRecognizor',
]
