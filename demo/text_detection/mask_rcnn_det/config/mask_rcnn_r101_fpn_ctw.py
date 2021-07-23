"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_r50.py
# Abstract       :    Model settings for mask-rcnn-based detector on Total-Text

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""
_base_ = "./mask_rcnn_r50_fpn_ctw.py"
model = dict(
    backbone=dict(depth=101)
)
checkpoint_config = dict(interval=5, filename_tmpl='checkpoint/res101_maskrcnn_ctw_epoch_{}.pth')
load_from = "/path/to/mask_rcnn_r101_fpn_1x_20181129-34ad1961.pth"