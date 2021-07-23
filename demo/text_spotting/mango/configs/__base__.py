"""
#################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __base__.py
# Abstract       :    Base model settings

# Current Version:    1.0.0
# Date           :    2020-06-24
#################################################################################################
"""
character = '../../datalist/character_list.txt'
num_grids = [40]
featmap_indices = (0, )
text_max_length = 25
type='SPOTTER'
model = dict(
    type='MANGO',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style='pytorch'
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
    ),
    centerline_seg_head=dict(
        type='CenterlineSegHead',
        in_channels=256,
        conv_out_channels=256,
        sigma=0.4,
        featmap_indices=featmap_indices,
        loss_seg=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
    ),
    grid_category_head=dict(
        type='GridCategoryHead',
        sigma=0.2,
        num_grids=num_grids,
        featmap_indices=featmap_indices,
        in_channels=256,
        conv_out_channels=256,
        num_classes=2,
        stacked_convs=2,
        loss_category=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
    multi_mask_att_head=dict(
        type='CharacterMaskAttentionHead',
        num_grids=num_grids,
        featmap_indices=featmap_indices,
        in_channels=256,
        conv_out_channels=256,
        stacked_convs=4,
        text_max_length=text_max_length,
        loss_char_mask_att=dict(
            type='DiceLoss',
            loss_weight=3.0
        ),
    ),
    attention_fuse_module=dict(
        type='AttFuseModule',
        featmap_indices=featmap_indices,
        in_channels=256,
        conv_out_channels=256,
        stacked_convs=4,
    ),
    semance_module=dict(
        type='CascadeRNN',
        rnn_modules=[
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True, ),
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True, )
        ],
    ),
    multi_recog_sequence_head=dict(
        type='MultiRecogSeqHead',
        text_max_length=text_max_length, # for [s] and [GO]
        featmap_indices=featmap_indices,
        in_channels=256,
        num_fcs=2,
        fc_out_channels=512,
        converter=dict(
            type='BertLabelConverter',
            character=character,
        ),
        loss_recog=dict(
            type='StandardCrossEntropyLoss',
            ignore_index=0,
            loss_weight=1.0,
            reduction='mean',
        ),
    ),
    test_cfg=dict(
        postprocess=dict(
            type='PostMango',
            seg_thr=0.5,
            cate_thr=0.5,
            do_visualization=False,
        ),
    ),
)

# dataset settings
dataset_type = 'TextSpotDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_bbox=False,           # Bounding Rect
         with_poly_mask=False,      # Mask
         with_poly_bbox=True,       # bouding poly
         with_label=False,          # Bboxes' labels
         with_care=True,            # Ignore or not
         with_text=True,            # Transcription
         with_cbbox=False,          # Character bounding
         text_profile=dict(text_max_length=text_max_length-1, sensitive="same", filtered=False)
         ),
    dict(type='DavarResize', img_scale=[(540, 720), (1440, 1800)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_poly_bboxes', 'gt_texts']),
]
test_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1440, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='DavarCollect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
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
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/path/to/datalist/test_datalist.json',
        img_prefix='/path/to/Image/',
        pipeline=test_pipeline))
# optimizer
find_unused_parameters = True
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[4, 7, 9])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(type="DavarCheckpointHook", interval=1, filename_tmpl='checkpoint/checkpoint_name_epoch_{}.pth')

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
load_from = '/path/to/Model_Zoo/resnet50-19c8e357.pth'
resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    model_type=type,
    type="DavarEvalHook",
    interval=1,
    eval_func_params=dict(
       IOU_CONSTRAINT=0.1,
       AREA_PRECISION_CONSTRAINT=0.1,
       WORD_SPOTTING=False
    ),
    by_epoch=True,
    eval_mode="general",
    save_best="hmean",
    rule='greater',
)