"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    lgpma_base.py
# Abstract       :    Base model settings for LGPMA detector
# Current Version:    1.0.1
# Date           :    2022-09-05
##################################################################################################
"""

model = dict(
    type='LGPMA',
    pretrained='path/to/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16],
            ratios=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='LGPMARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            # num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='LPMAMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=2,
            # num_classes=3,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            loss_lpma=dict(
                type='L1Loss', loss_weight=1.0))),
    global_seg_head=dict(
        type='GPMAMaskHead',
        in_channels=256,
        conv_out_channels=256,
        num_classes=1,
        loss_mask=dict(type='DiceLoss', loss_weight=1),
        loss_reg=dict(type='SmoothL1Loss', beta=0.1, loss_weight=0.01, reduction='sum')),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms_post=2000,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.5),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=1000,
            mask_thr_binary=0.5),
        postprocess=dict(
            type="PostLGPMA",
            refine_bboxes=False,  # If set to True, bboxes of aligned cells will be refined according to pyramid masks
        )
    ),
)

train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'TableRcgDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadTableAnnotations',
         with_bbox=True,        # Bounding Rect
         with_enlarge_bbox=True,# Produce pseudo-bboxes for aligned cells
         with_label=True,       # Bboxes' labels
         with_poly_mask=True,   # Mask
         with_empty_bbox=True,  # Produce pseudo-bboxes for empty cells
         ),
    dict(type='DavarResize', img_scale=[(360, 480), (960, 1080)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='GPMADataGeneration'),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.5,
        flip=False,
        transforms=[
            dict(type='DavarResize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DavarDefaultFormatBundle'),
            dict(type='DavarCollect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.5,
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
    samples_per_gpu=3,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='',
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='',
        img_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='',
        img_prefix='',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[6, 10])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, filename_tmpl='checkpoint/maskrcnn-lgpma-e{}.pth')
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# yapf:enable
# runtime settings

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = ''

load_from = None
resume_from = None
workflow = [('train', 1)]
