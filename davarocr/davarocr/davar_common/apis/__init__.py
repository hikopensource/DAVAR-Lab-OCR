"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from .inference import inference_model, init_model
from .test import single_gpu_test, multi_gpu_test
from .train import train_model
__all__ = [
    'inference_model',
    'train_model',
    'init_model',
    'single_gpu_test',
    'multi_gpu_test'
]
