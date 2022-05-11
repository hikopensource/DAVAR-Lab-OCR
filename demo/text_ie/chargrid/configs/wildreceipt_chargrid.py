"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    wildreceipt_chargrid.py
# Abstract       :    Model settings for chargrid-net-based extractor on wildreceipt

# Current Version:    1.0.0
# Date           :    2022-04-13
#########################################################################
"""
character = '/path/to/demo/text_ie/datalist/wildreceipt/flatten_dict.txt'
batch_max_length = 60
type = "SPOTTER"

model = dict(
    type='ChargridNetIE',
    # whether to use chargrid map as input, False to original image
    use_chargrid=True,  # for testing raw image input, set to False
    pretrained=None,
    backbone=dict(
        type='ChargridEncoder',
        input_channels=93,  # for testing raw image input, set to 3
        out_indices=(0, 1, 2, 4),
        base_channels=64),
    neck=dict(
        type='ChargridDecoder',
        base_channels=64),
    rpn_head=dict(
        type='RPNHead',
        in_channels=64,
        feat_channels=64,
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
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='ChargridRegHead',
            in_channels=64,
            conv_out_channels=64,
            roi_feat_size=7,
            num_classes=26,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9, loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=3,
            in_channels=64,
            conv_out_channels=64,
            num_classes=26,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    ie_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=(1, 1), sampling_ratio=0),
        out_channels=64,
        featmap_strides=[4, 8, 16, 32]
    ),
    ie_cls_head=dict(
        type='ClsHead',
        input_size=64,
        num_classes=26,  #
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False)
    ),
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
        # postprocess=dict(
        #     type="PostMaskRCNN"
        # )
    )
)

train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'WildReceiptDataset'
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
         with_text=True,
         text_profile=dict(text_max_length=batch_max_length, sensitive=True, filtered=False)
         ),
    dict(type='DavarResize', img_scale=[(512, 512)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ChargridDataGeneration', vocab=character, with_label=False, poly_shape=False),
    dict(type='ChargridFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'chargrid_map', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
val_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,           # Bounding Rect
         with_poly_mask=False,       # Mask
         with_label=False,          # Bboxes' labels
         with_care=True,
         with_text=True,
         text_profile=dict(text_max_length=batch_max_length, sensitive=True, filtered=False)
         ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ChargridDataGeneration', vocab=character, with_label=False, poly_shape=False),
            dict(type='ChargridFormatBundle'),
            dict(type='DavarCollect', keys=['img', 'chargrid_map', 'gt_bboxes']),
        ])
]
test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,  # Bounding Rect
         with_care=True,  # Ignore or not
         with_text=True,  # Transcription
         text_profile=dict(text_max_length=batch_max_length, sensitive=True, filtered=False)
         ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='DavarResize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ChargridDataGeneration',
                 visualize=False,             # Set true for visualization
                 #vis_save_dir='/path/to/demo/text_ie/chargrid/vis/wildreceipt/chargrid',
                 vocab=character,
                 with_label=False,
                 poly_shape=False),
            dict(type='ChargridFormatBundle'),
            dict(type='DavarCollect', keys=['img', 'chargrid_map', 'gt_bboxes']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/path/to/demo/text_ie/datalist/wildreceipt/Datalists/datalist_train.json',
        img_prefix='/path/to/wildreceipt',
        pipeline=train_pipeline,
        classes='/path/to/demo/text_ie/datalist/wildreceipt/class_list.txt'),
    val=dict(
        type=dataset_type,
        ann_file='/path/to/demo/text_ie/datalist/wildreceipt/Datalists/datalist_test.json',
        img_prefix='/path/to/wildreceipt',
        pipeline=val_pipeline,
        classes='/path/to/demo/text_ie/datalist/wildreceipt/class_list.txt'),
    test=dict(
        type=dataset_type,
        ann_file='/path/to/demo/text_ie/datalist/wildreceipt/Datalists/datalist_test.json',
        img_prefix='/path/to/wildreceipt',
        pipeline=test_pipeline,
        classes='/path/to/demo/text_ie/datalist/wildreceipt/class_list.txt')
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    step=[15, 30, 50])
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(type='DavarCheckpointHook', interval=1, save_mode='lightweight', metric='macro_f1',
                         filename_tmpl='checkpoint/res50_maskrcnn_epoch_{}.pth')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# yapf:enable
# runtime settings

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/path/to/demo/text_ie/chargrid/log/wildreceipt_chargrid'

load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    type="DavarDistEvalHook",
    model_type='SPOTTER',
    save_best='macro_f1',
    interval=1,
    metric='macro_f1',
    metric_options=dict(
        macro_f1=dict(
            ignores=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25])),
    rule='greater',
    priority='high'
)