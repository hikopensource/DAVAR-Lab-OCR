"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_r50_ctw.py
# Abstract       :    Model settings for mask-rcnn-based detector on ICDAR2015

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""
_base_ = "./__base__.py"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
val_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,           # Bounding Rect
         with_poly_mask=True,       # Mask
         with_label=True,          # Bboxes' labels
         with_care=True, ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1440),                # Testing scale for SCUT-CTW1500
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DavarDefaultFormatBundle'),
            dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels','gt_masks']),
        ])
]
test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1440),                # Testing scale for SCUT-CTW1500
        flip=False,
        transforms=[
            dict(type='DavarResize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DavarDefaultFormatBundle'),
            dict(type='DavarCollect', keys=['img']),
        ])
]

data = dict(
    train=dict(
        ann_file='/path/to/datalist/icdar2015_train_datalist.json',
        img_prefix='/path/to/ICDAR2015/'),
    val=dict(
        ann_file='/path/to/datalist/icdar2015_test_datalist.json',
        img_prefix='/path/to/ICDAR2015/',
        ),
    test=dict(
        ann_file='/path/to/datalist/icdar2015_test_datalist.json',
        img_prefix='/path/to/ICDAR2015/',
    ))

checkpoint_config = dict(interval=5, filename_tmpl='checkpoint/res50_maskrcnn_ic15_epoch_{}.pth')