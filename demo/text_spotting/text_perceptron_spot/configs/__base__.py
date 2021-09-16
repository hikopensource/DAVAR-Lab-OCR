# """
# #########################################################################
# # Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# # Filename       :    __base__.py
# # Abstract       :    Base model settings for text perceptron based text spotter.

# # Current Version:    1.0.0
# # Date           :    2021-09-15
# #########################################################################
# """
character = '../../datalist/character_list.txt'
batch_max_length = 32
type='SPOTTER'
model = dict(
    type='TextPerceptronSpot',
    # Pre-trained model, can be downloaded in the model zoo of mmdetection
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',),
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
        loss_reg_bond=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.01, reduction='sum'),),
    # rcg
    rcg_roi_extractor=dict(
        type='TPSRoIExtractor',
        in_channels=256,
        out_channels=256,
        point_num=14,
        output_size=(8, 32),
        featmap_strides=[4],),
    rcg_transformation=None,
    rcg_backbone=dict(
        type='LightCRNN',
        in_channels=256,
        out_channels=256
    ),
    rcg_neck=None,
    rcg_sequence_module=dict(
        type='CascadeRNN',
        rnn_modules=[
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True,),
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True,), ]),
    rcg_sequence_head=dict(
        type='AttentionHead',
        input_size=256,
        hidden_size=256,
        batch_max_length=batch_max_length,
        converter=dict(
            type='AttnLabelConverter',
            character=character,
            use_cha_eos=True,),
        loss_att=dict(          
            type='StandardCrossEntropyLoss',
            ignore_index=0,
            reduction='mean',
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        # rcg
        keep_dim=False),
    test_cfg=dict(
        # rcg
        keep_dim=False,
        batch_max_length=batch_max_length,
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
        )),
)
# training and testing settings
train_cfg = dict()
test_cfg = dict()

# Training dataset load type
dataset_type = 'DavarMultiDataset'

# File prefix path of the traning dataset
img_prefixes = [
    '/path/to/Image/'
]

# Dataset Name
ann_files = [
    '/path/to/datalist/train_datalist.json'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations',
         with_poly_bbox=True,       # bouding poly
         with_care=True,            # Ignore or not
         with_text=True,            # Transcription
         text_profile=dict(text_max_length=batch_max_length, sensitive='same', filtered=False)
    ),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DavarRandomCrop', instance_key='gt_poly_bboxes'),
    dict(type='RandomRotate', angles=[-15, 15], borderValue=(0, 0, 0)),
    dict(type='DavarResize', img_scale=[(736, 736)], multiscale_mode='value', keep_ratio=True),
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
    dict(type='DavarCollect', keys=['img', 'gt_texts', 'gt_masks', 'gt_poly_bboxes']),
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='DavarCollect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    sampler=dict(
        type='DistBatchBalancedSampler',  # BatchBalancedSampler and DistBatchBalancedSampler
        mode=0,
        # model 0:  Balance in batch, calculate the epoch according to the first iterative data set
        # model 1:  Balance in batch, calculate the epoch according to the last iterative data set
        # model 2:  Balance in batch, record unused data
        # model -1: Each dataset is directly connected and shuffled
    ),
    train=dict(
        type=dataset_type,
        batch_ratios=['1.0'],
        dataset=dict(
            type='TextSpotDataset',
            ann_file=ann_files,
            img_prefix=img_prefixes,
            test_mode=False,
            pipeline=train_pipeline)
    ),
    val=dict(
        type='TextSpotDataset',
        ann_file='/path/to/datalist/test_datalist.json',
        img_prefix='/path/to/Image/',
        pipeline=test_pipeline),
    test=dict(
        type='TextSpotDataset',
        ann_file='/path/to/datalist/test_datalist.json',
        img_prefix='/path/to/Image/',
        pipeline=test_pipeline))

# optimizer
find_unused_parameters = True
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    step=[2, 3]
)
runner = dict(type='EpochBasedRunner', max_epochs=4)
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
load_from = '/path/to/Model_Zoo/tp_r50_tt-5b348520.pth'
resume_from = None
workflow = [('train', 1)]

# # Online evaluation
evaluation = dict(
    model_type=type,
    type='DavarDistEvalHook',
    interval=1,
    eval_func_params=dict(
       # SPECIAL_CHARACTERS='[]+-#$()@=_!?,:;/.%&'\">*|<`{~}^\ ',
       IOU_CONSTRAINT=0.5,
       AREA_PRECISION_CONSTRAINT=0.5,
       WORD_SPOTTING=False
    ),
    by_epoch=True,
    eval_mode='general',
    # eval_mode='lightweight',
    save_best='hmean',
    rule='greater',
)

