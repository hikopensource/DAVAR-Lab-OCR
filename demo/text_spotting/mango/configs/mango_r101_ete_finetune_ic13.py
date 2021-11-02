"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango_r101_ete_finetune_ic13.py
# Abstract       :    Model settings for mango end-to-end train on realdata.

# Current Version:    1.0.0
# Date           :    2020-06-24
######################################################################################################
"""
_base_ = './mango_r50_ete_finetune_ic13.py'
model = dict(
    backbone=dict(depth=101)
)
checkpoint_config = dict(interval=10, filename_tmpl='checkpoint/res101_ete_finetune_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'
load_from = '/path/to/res101_ete_pretrain.pth'