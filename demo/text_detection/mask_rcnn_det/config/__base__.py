"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_r50.py
# Abstract       :    Model settings for mask-rcnn-based detector on Total-Text

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""
model = dict(
    type='MaskRCNNDet',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
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
            scales=[8],
            ratios=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
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
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
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
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
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
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100,
            mask_thr_binary=0.5),
        postprocess=dict(
            type="PostMaskRCNN"
        )),
)

train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'TextDetDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,           # Bounding Rect
         with_poly_mask=True,       # Mask
         with_label=True,          # Bboxes' labels
         with_care=True,            # Ignore or not
         ),
    dict(type='Resize', img_scale=[(720, 640), (1800, 1600)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels','gt_masks']),
]
val_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,           # Bounding Rect
         with_poly_mask=True,       # Mask
         with_label=True,          # Bboxes' labels
         with_care=True, ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1350, 950),                # Testing scale for Total-Text
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
        img_scale=(1350, 950),                # Testing scale for Total-Text
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/path/to/datalist/train_datalist.json',
        img_prefix='/path/to/Image/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/path/to/datalist/test_datalist.json',
        img_prefix='/path/to/Image/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/path/to/datalist/test_datalist.json',
        img_prefix='/path/to/Image/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[15, 30, 45])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5, filename_tmpl='checkpoint/res50_maskrcnn_epoch_{}.pth')
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
work_dir = '/path/to/workspace/log/'

load_from = "/path/to/Model_Zoo/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth"
resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    type="DavarDistEvalHook",
    interval=1,
    eval_func_params=dict(
        IOU_CONSTRAINT=0.5,
        AREA_PRECISION_CONSTRAINT=0.5,
        CONFIDENCES=False,
    ),
)