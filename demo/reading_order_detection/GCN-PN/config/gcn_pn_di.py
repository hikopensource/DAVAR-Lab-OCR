# model settings
batch_max_length = 145
type="SPOTTER"
model = dict(
    type='GCN_PN',
    pretrained="/path/to/resnext101_64x4d-ee2c6f71.pth",
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
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
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),),
    infor_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=(1, 1), sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),#
    infor_context_module=dict(
        type='GraphConvEncoder',
        in_channel = 256*3,
        output_channel = 256,
        graph_conv_block_num=2),
    infor_node_cls_head=dict(
        type='PointerHead',
        init_channel=256,
        query_in_channel=256,
        query_out_channel=256,
        key_in_channel=256,
        key_out_channel=256,
    ),
    

    # model training and testing settings
    train_cfg=dict(
        # rcg
        keep_dim=False,
        sequence=(),

        # infor
        down_sample_first=False,

        reproduce_labels=False,
        batch_max_length=batch_max_length,

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
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        # rcg
        keep_dim=False,
        sequence=dict(),
        batch_max_length=batch_max_length,

        # info
        down_sample_first=False,

        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

# dataset settings
dataset_type = 'DIOrderDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DavarLoadImageFromFile',),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,           # Bounding Rect
         with_poly_mask=False,      # Mask
         with_poly_bbox=False,       # bouding poly
         with_label=True,          # Bboxes' labels
         with_care=True,            # Ignore or not
         ),
    dict(type='DavarResize', img_scale=[(512, 512)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='DavarLoadImageFromFile',),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,  # Bounding Rect
         with_poly_mask=False,  # Mask
         with_poly_bbox=False,  # bouding poly
         with_label=True,  # Bboxes' labels
         with_care=True,  # Ignore or not
         ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DavarDefaultFormatBundle'),
            dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
test_pipeline = [
    dict(type='DavarLoadImageFromFile',),
    dict(type='DavarLoadAnnotations',
         with_bbox=True,  # Bounding Rect
         with_poly_mask=False,  # Mask
         with_poly_bbox=False,  # bouding poly
         with_label=True,  # Bboxes' labels
         with_care=True,  # Ignore or not
         ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DavarDefaultFormatBundle'),
            dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file="/path/to/train_6175_wo_0.json",
        img_prefix="/path/to/DI_dataset/Images/",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file="/path/to//val_1300_wo_0.json",
        img_prefix="/path/to/DI_dataset/Images/",
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file="/path/to/test_1300_wo_0.json",
        img_prefix="/path/to/DI_dataset/Images/",
        pipeline=test_pipeline
    ))
# optimizer
find_unused_parameters = True
optimizer = dict(type='AdamW', lr=1e-3)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='inv',
                 warmup='linear',
                 warmup_iters=10,
                 warmup_ratio=0.00001,
                 warmup_by_epoch=True,
                 gamma=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=50)

checkpoint_config = dict(type='DavarCheckpointHook', interval=1, save_mode='lightweight', metric='total_order_acc',
                         filename_tmpl='checkpoint/DI_{}.pth', save_last=False)
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
work_dir = '/path/to/work_dir/log/'
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    save_best='total_order_acc',
    model_type='SPOTTER',
    interval=1,
    metric='total_order_acc',
    rule="greater",
    priority='HIGH')
