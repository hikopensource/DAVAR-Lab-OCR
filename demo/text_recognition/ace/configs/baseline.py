"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    baseline.py
# Abstract       :    ACE recognition Model Baseline

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""

# encoding=utf-8
_base_ = [
    '../../__base__/res32_bilstm_attn.py'
]

work_dir = '/data1/workdir/davar_opensource/ace/'
