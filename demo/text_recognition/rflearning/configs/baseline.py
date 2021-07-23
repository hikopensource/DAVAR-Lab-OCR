"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    baseline.py
# Abstract       :    RF-Learning recognition Model Baseline

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""

# encoding=utf-8
_base_ = [
    '../../__base__/res32_bilstm_attn.py'
]

"""
4. runtime setting
description:
    Set the number of training epochs and working directory according to the actual situation

Add keywords:
    None
"""

# work_directory
work_dir = '/data1/workdir/davar_opensource/rflearning/'
