"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_r50_conv6_e2e_pretrain.py
# Abstract       :    Model settings for mask rcnn spotter end-to-end pretrain on synthdata.

# Current Version:    1.0.0
# Date           :    2021-09-15
######################################################################################################
"""
_base_ = './__base__.py'

# File prefix path of the traning dataset
img_prefixes = [
    '/path/to/SynthText/',
    '/path/to/COCO-Text/',
]

# Dataset Name
ann_files = [
    '/path/to/datalist/synthtext_80w.json',
    '/path/to/datalist/cocotext.json'
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    sampler=dict(
        type='DistBatchBalancedSampler',  # BatchBalancedSampler and DistBatchBalancedSampler
        mode=1,
        # model 0:  Balance in batch, calculate the epoch according to the first iterative data set
        # model 1:  Balance in batch, calculate the epoch according to the last iterative data set
        # model 2:  Balance in batch, record unused data
        # model -1: Each dataset is directly connected and shuffled
    ),
    train=dict(
        batch_ratios=['0.5', '0.5'],
        dataset=dict(
            ann_file=ann_files,
            img_prefix=img_prefixes,
        )
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
checkpoint_config = dict(interval=1, filename_tmpl='checkpoint/mask_rcnn_r50_conv6_e2e_pretrain_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'
load_from = '/path/to/Model_Zoo/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth'
