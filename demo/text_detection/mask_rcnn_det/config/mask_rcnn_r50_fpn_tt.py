"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_r50.py
# Abstract       :    Model settings for mask-rcnn-based detector on Total-Text

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""
_base_ = "./__base__.py"
data = dict(
    train=dict(
        ann_file='/path/to/datalist/total_text_train_datalist.json',
        img_prefix='/path/to/Total-Text/',
    ),
    val=dict(
        ann_file='/path/to/datalist/total_text_test_datalist.json',
        img_prefix='/path/to/Total-Text/',
        ),
    test=dict(
        ann_file='/path/to/datalist/total_text_test_datalist.json',
        img_prefix='/path/to/Total-Text/',
    ))

checkpoint_config = dict(interval=5, filename_tmpl='checkpoint/res50_maskrcnn_tt_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'

