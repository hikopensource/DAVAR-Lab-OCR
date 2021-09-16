"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_r50_e2e_finetune.py
# Abstract       :    Model settings for text perceptron spotter end-to-end finetune on realdata.

# Current Version:    1.0.0
# Date           :    2021-09-15
######################################################################################################
"""
_base_ = './__base__.py'

# File prefix path of the traning dataset
img_prefixes = [
    '/path/to/ICDAR2013-Focused-Scene-Text/',
    '/path/to/ICDAR2015/',
    '/path/to/ICDAR2017_MLT/',
    '/path/to/Total-Text/',
]

# Dataset Name
ann_files = [
    '/path/to/datalist/icdar2013_train_datalist.json',
    '/path/to/datalist/icdar2015_train_datalist.json',
    '/path/to/datalist/icdar2017_trainval_datalist.json',
    '/path/to/datalist/total_text_train_datalist.json',
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    sampler=dict(
        type='DistBatchBalancedSampler',  # BatchBalancedSampler and DistBatchBalancedSampler
        mode=1,
        # model 0:  Balance in batch, calculate the epoch according to the first iterative data set
        # model 1:  Balance in batch, calculate the epoch according to the last iterative data set
        # model 2:  Balance in batch, record unused data
        # model -1: Each dataset is directly connected and shuffled
    ),
    train=dict(
        batch_ratios=['0.25', '0.25', '0.25', '0.25'],
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
optimizer=dict(lr=1e-4)
lr_config = dict(step=[20, 30])
runner = dict(max_epochs=40)
checkpoint_config = dict(interval=10, filename_tmpl='checkpoint/tp_r50_e2e_finetune_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'
load_from = '/path/to/tp_r50_e2e_pretrain.pth'
evaluation = dict(
    interval=10,
)
