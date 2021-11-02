"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango_r101_ete_finetune_ic15.py
# Abstract       :    Model settings for mango end-to-end train on realdata.

# Current Version:    1.0.0
# Date           :    2020-06-24
######################################################################################################
"""
_base_ = './mango_r101_ete_finetune_ic13.py'
model = dict(
    test_cfg=dict(
        postprocess=dict(
            seg_thr=0.5,
            cate_thr=0.5,
        ),
    ),
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1800, 1600),
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