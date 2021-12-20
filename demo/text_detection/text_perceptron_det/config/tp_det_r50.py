"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_det_r50.py
# Abstract       :    Model settings, base setting of in resnet50 backbone

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""
model = dict(
    type='TextPerceptronDet',
    # Pre-trained model, can be downloaded in the model zoo of mmdetection
    pretrained='/path/to/pretrained_model/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_input",
        num_outs=4),
    mask_head=dict(
        type='TPHead',
        in_channels=256,
        conv_out_channels=256,
        conv_cfg=None,
        norm_cfg=None,
        # All of the segmentation losses, including center text/ head/ tail/ top&bottom boundary
        loss_seg=dict(type='DiceLoss', loss_weight=1.0),
        # Corner regression in head region
        loss_reg_head=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.1, reduction='sum'),
        # Corner regression in tail region
        loss_reg_tail=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.1, reduction='sum'),
        # boundary offset regression in center text region
        loss_reg_bond=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.01, reduction='sum'),
        ),

)
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    postprocess=dict(
        type='TPPointsGeneration',
        # Re-implenmented in C++ (You can implement it in CUDA for further speed up), comment to use default one
        # lib_name='tp_points_generate.so',
        # lib_dir='/path/to/davarocr/davar_det/core/post_processing/lib/'),
        # Parameters for points generating
        filter_ratio=0.6,
        thres_text=0.35,
        thres_head=0.45,
        thres_bond=0.35,
        point_num=14
    )
)

dataset_type = 'TextDetDataset'

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
        shrink_head_ratio=0.25,
        shrink_bond_ratio=0.09,
        ignore_ratio=0.6),
    dict(type='SegFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_masks'],
         meta_keys=('filename', 'scale_factor', 'pad_shape')),
]

test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1350, 950),                # Testing scale for Total-Text
        flip=False,
        transforms=[
            dict(type='DavarResize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DavarDefaultFormatBundle'),
            dict(type='DavarCollect', keys=['img'],meta_keys=('filename', 'scale_factor', 'pad_shape')),
        ])
]
find_unused_parameters = True
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,

    train=dict(
        type=dataset_type,
        ann_file=[
            '/path/to/datalist/train_datalist.json'
                  ],
        img_prefix=[
            '/path/to/Images/'
                ],
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        ann_file='/path/to/datalist/test_datalist.json',
        img_prefix='/path/to/Images/',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        samples_per_gpu=1,
        ann_file='/path/to/datalist/test_datalist.json',
        img_prefix='/path/to/Images/',
        pipeline=test_pipeline,
        test_mode=True
        ))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8, 16, 24])
total_epochs = 30

# checkpoint saved path
checkpoint_config = dict(interval=10,  filename_tmpl='checkpoint/tp_r50_3stage_epoch_{}.pth')
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# runtime settings
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'

# Work space to save training log
work_dir = '/path/to/workspace/log/'

# 'Pretrained model on Synthtext'
# you can simply load_from = 'path/to/tp_det_r50_tt_e25-45b1f5cf.pth' to fine-tune current model into a new domain
load_from = "/path/to/workspace/log/checkpoint/tp_r50_pretrained_synthtextmix-f45527e8.pth"

resume_from = None
workflow = [('train', 1)]

# Online evaluation
evaluation = dict(
    type="DavarDistEvalHook",
    interval=5,
    eval_func_params=dict(
        IOU_CONSTRAINT=0.5,
        AREA_PRECISION_CONSTRAINT=0.5,
        CONFIDENCES=False,
    ),
)