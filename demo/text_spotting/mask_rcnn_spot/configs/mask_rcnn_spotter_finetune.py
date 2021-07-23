"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango_r50_ete_pretrain.py
# Abstract       :    Model settings for mask rcnn spotter end-to-end finetune on realdata.

# Current Version:    1.0.0
# Date           :    2020-06-24
######################################################################################################
"""
_base_ = './__base__.py'
batch_max_length = 25

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,            # Bounding Rect
         with_poly_mask=True,       # Mask
         with_poly_bbox=True,      # bouding poly
         with_label=True,           # Bboxes' labels
         with_care=True,            # Ignore or not
         with_text=True,            # Transcription
         with_cbbox=False,          # Character bounding
         text_profile=dict(text_max_length=batch_max_length, sensitive="same", filtered=False)
    ),
    dict(type='RandomRotate', angles=[-15, 15], borderValue=(0, 0, 0)),
    dict(type='DavarResize', img_scale=[(640, 720), (1600, 1800)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_texts', 'gt_masks']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_file=[
            '/path/to/datalist/icdar2013_train_datalist.json',
            '/path/to/datalist/icdar2015_train_datalist.json',
            '/path/to/datalist/icdar2017_trainval_datalist.json',
            '/path/to/datalist/icdar2019_train_datalist.json',
            '/path/to/datalist/total_text_train_datalist.json',
        ],
        img_prefix=[
            '/path/to/ICDAR2013-Focused-Scene-Text/',
            '/path/to/ICDAR2015/',
            '/path/to/ICDAR2017_MLT/',
            '/path/to/ICDAR2019_MLT/',
            '/path/to/Total-Text/',
        ],
        pipeline=train_pipeline
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
lr_config = dict(step=[30, 50])
runner = dict(max_epochs=60)
checkpoint_config = dict(interval=10, filename_tmpl='checkpoint/res50_ete_finetune_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'
load_from = '/path/to/res50_ete_pretrain.pth'
evaluation = dict(
    interval=10,
)
