"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    east_r50_rbox.py
# Abstract       :    Model settings for EAST detector on ICDAR2015(RBOX mode)

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""
# model_setting
model = dict(
    type='SpatialTempoEASTDet',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        ),
    neck=dict(
        type='EastMerge',
        in_channels=[256, 512, 1024, 2048],
    ),
    mask_head=dict(
        type='SpatialTempoEASTHead',
        loss_seg=dict(type='DiceLoss', loss_weight=0.01),
        loss_reg=dict(type='EASTIoULoss', loss_weight=1, mode='iou'),# Support 'RBOX' only
        geometry='RBOX',
        window_size=5),
    test_cfg=dict(
        postprocess=dict(
            type='PostEAST',
            thres_text=0.9,
            nms_thres=0.2,
            nms_method='RBOX'
        )
    )
)

# Model training and test parameter configuration
train_cfg = dict(                # Dimensions remain or change
    fix_backbone=True
)


# dataset settings
dataset_type = 'MultiFrameDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='ConsistLoadImageFromFile'),
    dict(type='ConsistLoadAnnotations', with_poly_bbox=True, with_care=True),
    dict(type='ConsistRandomRotate', angles=(-15, 15)),
    dict(type='ConsistResize', img_scale=[(768, 512), (1920, 1080)], multiscale_mode='range', keep_ratio=True),
    dict(type='ConsistColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    dict(type='ConsistNormalize', **img_norm_cfg),
    dict(type='ConsistPad', size_divisor=32),
    dict(type='EASTDataGeneration', geometry='RBOX'),
    dict(type='ConsistFormatBundle'),
    dict(type='ConsistCollect', keys=['img', 'gt_masks'], meta_keys=('filename', 'scale_factor', 'pad_shape', 'video',
                                                                     'frameID', 'flow')),
]

test_pipeline = [
    dict(type='ConsistLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type='ConsistResize', keep_ratio=True),
            dict(type='ConsistNormalize', **img_norm_cfg),
            dict(type='ConsistPad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ConsistCollect', keys=['img'], meta_keys=('filename', 'scale_factor', 'pad_shape', 'video',
                                                                 'frameID', 'flow', 'pre_features')),
        ])
]

find_unused_parameters = True

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    cfg_collate='YORO_collate',
    train=dict(
        type=dataset_type,
        window_size=5,
        ann_file=[
            "/path/to/demo/videotext/datalist/video_ic15_train_datatlist_filterLess3_quality.json",
        ],
        img_prefix=[
            "/path/to/VideoText/IC15/ch3_train/",
        ],
        pipeline=train_pipeline,
        flow_path='/path/to/ic15_train_flow/',
        ),
    val=dict(
        type=dataset_type,
        ann_file='/path/to/demo/videotext/datalist/ic13_video_test_datalist.json',
        img_prefix='/path/to/VideoText/IC13/',
        pipeline=test_pipeline
        ),
    test=dict(
        type=dataset_type,
        ann_file='/path/to/demo/videotext/datalist/ic13_video_test_datalist.json',
        img_prefix='/path/to/VideoText/IC13/',
        pipeline=test_pipeline
        )
)

# optimizer
optimizer = dict(type='Adam', lr=0.7)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[200, 400])
runner = dict(type='IterBasedRunner', max_iters=60000)

f_name = 'SR_east_res50_ic15_rbox'
checkpoint_config = dict(interval=1000, filename_tmpl='checkpoint/'+f_name+'_fix_iter{}.pth')
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# yapf:enable
# runtime settings
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/path/to/workspace/ic15_yoro_east_det_fix_mask0.3/log/'

load_from = '/path/to/base_east_ic15_mix/log/checkpoint/SR_east_res50_ic15_rbox_epoch170.pth'

resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    eval_func_params=dict(
        IOU_CONSTRAINT=0.5,
        AREA_PRECISION_CONSTRAINT=0.5,
    ),
    by_epoch=True,
    interval=10,
    eval_mode="general",
    save_best="hmean",
    rule='greater',
)