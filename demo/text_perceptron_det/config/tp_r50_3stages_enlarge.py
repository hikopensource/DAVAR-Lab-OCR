"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_r50_3stages_enlarge.py
# Abstract       :    Model settings

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31
#########################################################################
"""
import os

# Absolute path to the 'third_party' directory
third_party_path = '/path/to/mmdetection/third_party/'

model = dict(
    type='TextPerceptronDet',
    # Pre-trained model, can be downloaded in the model zoo of mmdetection
    pretrained='/path/to/pretrained_model/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 2, 1),   # enlarge receptive field in 8x feature map
        strides=(1, 1, 2),
        frozen_stages=-1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=3),
    mask_head=dict(
        type='TPHead',
        in_channels=256,
        conv_out_channels=256,
        conv_cfg=None,
        norm_cfg=None,
        # All of the segmentation losses, including center text/ head/ tail/ top&bottom boundary
        loss_seg=dict(type='DiceLoss', loss_weight=1.0),
        # Corner regression in head region
        loss_reg_head=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.1,
                           reduction='sum'),
        # Corner regression in tail region
        loss_reg_tail=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.1,
                           reduction='sum'),
        # boundary offset regression in center text region
        loss_reg_bond=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.01,
                           reduction='sum'),
        ),
    shape_transform_module=dict(
        type='PointsGeneration',
        # Re-implenmented in C++ (You can implement it in CUDA for further speed up)
        libname='tp_points_generate.so',
        libdir=os.path.join(third_party_path,
                            'text_perceptron/mmdet/models/shape_transform_module/lib/'),
        # Parameters for points generating
        filter_ratio=0.6,
        thres_text=0.35,
        thres_head=0.45,
        thres_bond=0.35,
        point_num=14
    )
)
# training and testing settings
train_cfg = dict()
test_cfg = dict()

dataset_type = 'CustomDataset'
data_root = ''

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # Implemented in third_party/text_perceptron
    dict(type='RandomRotate', angles=(-15, 15)),
    dict(type='DavarResize', img_scale=[(640, 960), (1400, 1600)],
         multiscale_mode='range', keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # Ground truth generation
    dict(type='TPDataGeneration',
        shrink_head_ratio=0.35,
        shrink_bond_ratio=0.15,
        ignore_ratio=0.6,
        lib_name='tp_data.so',
        lib_dir=os.path.join(third_party_path,
                             'text_perceptron/mmdet/datasets/pipelines/lib/')),
    dict(type='TPFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks'],
         meta_keys=('filename', 'scale_factor', 'pad_shape')),
]
test_pipeline = [
    dict(type='LoadTestImg'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1250, 950),                # Testing scale for SCUT-CTW1500
        img_scale=(1350, 950),                # Testing scale for Total-Text
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[
                '/path/to/datalist/total_text_train_datalist.json'
                #'/path/to/datalist/ctw1500_train_datalist.json'
                  ],
        img_prefix=[
                '/path/to/Images/Total-Text/'
                #'/path/to/Images/SCUT-CTW1500/'
                ],
        pipeline=train_pipeline
        ),
    # Not used
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        imgs_per_gpu=1,
        ann_file=None,
        img_prefix=None,
        pipeline=test_pipeline
        ))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.00005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 3,
    step=[8, 16, 24])

# checkpoint saved path
checkpoint_config = dict(interval=5,  filename_tmpl='checkpoint/tp_r50_tt_epoch_{}.pth')
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 30
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'

# Work space to save training log
work_dir = 'path/to/save/log/'

# 'Pretrained model on Synthtext'
# you can simply load_from = 'path/to/tp_det_r50_tt_e25-45b1f5cf.pth' to fine-tune current model into a new domain
load_from = None

resume_from = None
workflow = [('train', 1)]
