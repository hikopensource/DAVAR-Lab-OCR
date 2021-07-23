"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_r50_3stages_enlarge_ctw.py
# Abstract       :    Model settings for ctw1500

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""

_base_= "./tp_det_r50.py"
model = dict(
    backbone=dict(
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 2, 1),   # enlarge receptive field in 8x feature map
        strides=(1, 1, 2)
    ),
    neck=dict(
        in_channels=[256, 512, 1024],
        num_outs=3)
)

data = dict(
    samples_per_gpu=2,
    train=dict(
        ann_file=[
            '/path/to/datalist/ctw1500_train_datalist.json',
            ],
        img_prefix=[
            '/path/to/SCUT-CTW1500/',
            ],
        ),
  )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations', with_poly_bbox=True),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='RandomRotate', angles=(-15, 15)),
    dict(type='DavarResize', img_scale=[(640, 960), (1400, 1600)], multiscale_mode='range', keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # Ground truth generation
    dict(type='TPDataGeneration',
        # Comment to use default setting
        # lib_name='tp_data.so',
        # lib_dir='/path/to/davarocr/davar_det/datasets/pipelines/lib/'),
        shrink_head_ratio=0.35,
        shrink_bond_ratio=0.15,
        ignore_ratio=0.6),
    dict(type='SegFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks'],
         meta_keys=('filename', 'scale_factor', 'pad_shape')),
]

# checkpoint saved path
checkpoint_config = dict(interval=5,  filename_tmpl='checkpoint/tp_r50_3stages_enlarge_ctw_epoch_{}.pth')

# 'Pretrained model on Synthtext'
# you can simply load_from = 'path/to/tp_det_r50_tt_e25-45b1f5cf.pth' to fine-tune current model into a new domain
load_from = "/path/to/workspace/log/checkpoint/tp_det_r50_3stages_enlarge_ctw-c1bf44e7.pth"
