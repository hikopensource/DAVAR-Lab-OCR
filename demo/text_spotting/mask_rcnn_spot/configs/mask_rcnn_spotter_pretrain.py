"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango_r50_ete_pretrain.py
# Abstract       :    Model settings for mask rcnn spotter end-to-end pretrain on synthdata.

# Current Version:    1.0.0
# Date           :    2020-06-24
######################################################################################################
"""
_base_ = './__base__.py'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_file=[
            '/path/to/datalist/synthtext_80w.json',
        ],
        img_prefix=[
            '/path/to/SynthText/',
        ]
    ),
    val=dict(
        ann_file='/path/to/datalist/icdar2013_test_datalist.json',
        img_prefix='/path/to/ICDAR2013-Focused-Scene-Text/',
    ),
    test=dict(
        ann_file='/path/to/datalist/icdar2013_test_datalist.json',
        img_prefix='/path/to/ICDAR2013-Focused-Scene-Text/',
    )
)
optimizer=dict(lr=1e-3)
lr_config = dict(step=[2, 3])
runner = dict(max_epochs=4)
checkpoint_config = dict(interval=1, filename_tmpl='checkpoint/res50_ete_pretrain_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'
load_from = '/path/to/Model_Zoo/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth'
