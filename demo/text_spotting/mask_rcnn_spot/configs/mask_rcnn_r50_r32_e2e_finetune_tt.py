"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_r50_r32_e2e_finetune_tt.py
# Abstract       :    Model settings for mask rcnn spotter end-to-end finetune on realdata.

# Current Version:    1.0.0
# Date           :    2021-06-24
######################################################################################################
"""
_base_ = "./mask_rcnn_r50_r32_e2e_finetune_ic13.py"

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1350, 950),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='DavarCollect', keys=['img']),
        ])
]

data = dict(
    test=dict(
        pipeline=test_pipeline
    )
)
