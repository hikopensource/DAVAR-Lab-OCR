"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango_r101_att_pretrain.py
# Abstract       :    Model settings for mango attention pretrain on synthdata.

# Current Version:    1.0.0
# Date           :    2020-06-24
######################################################################################################
"""
_base_ = './mango_r50_att_pretrain.py'
model = dict(
    backbone=dict(depth=101)
)
checkpoint_config = dict(interval=1, filename_tmpl='checkpoint/res101_att_pretrain_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'
load_from = '/path/to/Model_Zoo/resnet101-5d3b4d8f.pth'