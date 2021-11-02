"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango_r50_ete_finetune_ic13.py
# Abstract       :    Model settings for mango end-to-end train on realdata.

# Current Version:    1.0.0
# Date           :    2020-06-24
######################################################################################################
"""
_base_ = './__base__.py'

text_max_length = 25

model = dict(
    multi_mask_att_head=dict(
        loss_char_mask_att=None
    ),
    test_cfg=dict(
        postprocess=dict(
            seg_thr=0.1,
            cate_thr=0.1,
        ),
    ),
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=False,           # Bounding Rect
         with_poly_mask=False,      # Mask
         with_poly_bbox=True,       # bouding poly
         with_label=False,          # Bboxes' labels
         with_care=True,            # Ignore or not
         with_text=True,            # Transcription
         with_cbbox=False,          # Character bounding
         text_profile=dict(text_max_length=text_max_length-1, sensitive="same", filtered=False)
    ),
    dict(type='RandomRotate', angles=(-15, 15), borderValue=(0, 0, 0)),
    dict(type='DavarResize', img_scale=[(540, 720), (1440, 1800)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_poly_bboxes', 'gt_texts']),
]

test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1440, 960),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
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
        pipeline=test_pipeline
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
