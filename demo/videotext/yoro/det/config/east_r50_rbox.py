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
    type='EAST',
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
        type='EASTHead',
        loss_seg=dict(type='DiceLoss', loss_weight=0.01),
        loss_reg=dict(type='EASTIoULoss', loss_weight=1, mode='iou'),   # Support 'RBOX' only
        geometry='RBOX'),
    train_cfg=dict(),
    test_cfg=dict(
        postprocess=dict(
            type='PostEAST',
            thres_text=0.9,
            nms_thres=0.2,
            nms_method='RBOX'
        )
    )
)

# dataset settings
dataset_type = 'TextDetDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations', with_poly_bbox=True, with_care=True),
    dict(type='RandomRotate', angles=(-15, 15), borderValue=(0, 0, 0)),
    dict(type='DavarResize', img_scale=[(768, 512), (1920, 1080)], multiscale_mode='range', keep_ratio=True),
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='EASTDataGeneration', geometry='RBOX'),
    dict(type='SegFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_masks']),
]

test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type='DavarResize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='DavarCollect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        ann_file=[
            "/path/to/demo/videotext/datalist/video_ic15_train_datatlist_filterLess3_quality.json",
            "/path/to/demo/text_detection/datalist/icdar2015_train_datalist.json"
        ],
        img_prefix=[
            "/path/to/VideoText/IC15/ch3_train/",
            "/path/to/TextDetection/ICDAR2015/"
        ],
        pipeline=train_pipeline
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
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[200, 400])
runner = dict(type='EpochBasedRunner', max_epochs=600)

f_name = 'SR_east_res50_ic15_rbox'
checkpoint_config = dict(interval=10, by_epoch=True, filename_tmpl='checkpoint/'+f_name+'_epoch{}.pth')
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
work_dir = '/path/to/workspace/base_east_ic15_mix/log/'
load_from = None
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