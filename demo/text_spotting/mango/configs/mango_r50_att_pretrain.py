"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango_r50_att_pretrain.py
# Abstract       :    Model settings for mango attention pretrain on synthdata.

# Current Version:    1.0.0
# Date           :    2020-06-24
######################################################################################################
"""
_base_ = './__base__.py'

text_max_length = 25
model = dict(
    attention_fuse_module=None,
    semance_module=None,
    multi_recog_sequence_head=None,
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
         with_cbbox=True,           # Character bounding
         text_profile=dict(text_max_length=text_max_length-1, sensitive="same", filtered=False)
    ),
    dict(type='DavarResize', img_scale=[(540, 720), (1440, 1800)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_poly_bboxes', 'gt_texts', 'gt_cbboxes']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file='/path/to/datalist/synthtext_with_char.json',
        img_prefix='/path/to/SynthText/',
        pipeline=train_pipeline,
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
optimizer=dict(lr=1e-2)
lr_config = dict(step=[2, 3, 4])
runner = dict(max_epochs=5)
checkpoint_config = dict(interval=1, filename_tmpl='checkpoint/res50_att_pretrain_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'
load_from = '/path/to/Model_Zoo/resnet50-19c8e357.pth'
