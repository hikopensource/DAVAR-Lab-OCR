"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    res32_bilstm_attn.py
# Abstract       :    Base recognition Model, res32 bilstm attn

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""
# encoding=utf-8
_base_ = [
]

"""
1. model setting
description:
    Text recognition model configuration information

Add keywords:
    None
"""

# model setting
type = 'RECOGNIZOR'

# recognition dictionary
character = "/path/to/demo/text_recognition/__dictionary__/Scene_text_36.txt"

# Model setting
model = dict(
    type='TextRecommender',
    pretrained=None,
    backbone=dict(              # Backbone parameter
        type='CustomResNet32',
        input_channel=3,
        output_channel=512,),
    transformation=None,
    neck=None,                  # Relation module parameter
    sequence_module=dict(       # Sequential module parameter
        type='CascadeRNN',
        rnn_modules=[
            dict(
                type='BidirectionalLSTM',
                input_size=512,
                hidden_size=256,
                output_size=256,
                with_linear=True,
                bidirectional=True,),
            dict(
                type='BidirectionalLSTM',
                input_size=256,
                hidden_size=256,
                output_size=512,
                with_linear=True,
                bidirectional=True,), ]),
    sequence_head=dict(          # Recognition head parameter
        type='TextRecommenderHead',
        input_size=512,
        hidden_size=256,
        batch_max_length=25,
        converter=dict(          # Recognition Converter parameter
            type='AttnLabelConverter',
            character=character,
            use_cha_eos=True,),
        loss_att=dict(
            type='StandardCrossEntropyLoss',
            ignore_index=0,
            reduction='mean',
            loss_weight=1.0)
        )
)

find_unused_parameters = True

# Model training and test parameter configuration
# Model training and test parameter configuration
train_cfg = dict(                # Dimensions remain or change
    sequence=dict(),
    keep_dim=False,
    fix_rcg=False,
    fix_track=True,
    fix_qscore=True
)

test_cfg = dict(
    sequence=dict(),
    keep_dim=False,
    batch_max_length=25,
)


"""
2. Data Setting
description:
    Pipeline and training dataset settings

Add keywords:
    None
"""

# dataset settings
# support the dataset type
ppld = {
    'LMDB_Standard': 'LoadImageFromLMDB',  # open-source LMDB data

    # Davar dataset type
    'LMDB_Davar': 'RCGLoadImageFromLMDB',
    'File': 'RCGLoadImageFromFile',
    'Loose': 'RCGLoadImageFromLoose',
    'Tight': 'RCGLoadImageFromTight',
}

"""
Dataset Instruction manual:

data_types=['LMDB','File','Tight','File']     # corresponding to different data type

ann_files = ['train1|train2|train3',
             'Datalist/train1.json|Datalist/train2.json', 
             'Datalist/train_xxx.json',
             'Datalist/train_yyy.json']       # Separated by '|'

img_prefixes = ['xx/yy/zz/|aa/bb/cc/|mm/nn/', 
                 'dd/ee/', 'ff/gg/hh/', 
                 'ii/jj/kk/']                 # Separated by '|', corresponding to the ann_files

batch_ratios = ['0.1|0.1|0.1',
                '0.2|0.2',
                '0.1',
                '0.2']                        # string format, corresponding to the ann_files
                                              # sum of the batch_ratios equals to 1
"""

# Training dataset format
data_types = [
    'File',
]

# File prefix path of the traning dataset
img_prefixes = [
    '/path/to/VideoText/IC15/ch3_train',
]


# Dataset Name
ann_files = [
    '/path/to/demo/videotext/datalist/video_ic15_train_datatlist_filterLess3_quality.json'
]

# Training dataset load type
dataset_type = 'DavarMultiDataset'

# Normalization parameter
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5])

# training pipeline parameter
train_pipelines = [
    dict(
        type=ppld["File"],
        character=character,        # recognition dictionary
        test_mode=False,            # whether is in test mode
        sensitive=False,            # sensitive to Upper or Lower
        color_types=["rgb"],       # color loading type, ["rgb", "bgr", "gray"]
        fil_ops=True,
    ),
    dict(
        type='ResizeNormalize',
        size=(100, 32),
        interpolation=2,
        # Interpolation method of the Resize function
        # 0 - INTER_NEAREST(default)   # 1 - INTER_LINEAR
        # 2 - INTER_CUBIC              # 3 - INTER_AREA
        mean=img_norm_cfg["mean"],
        std=img_norm_cfg["std"], ),
    dict(type='DavarDefaultFormatBundle'),                # Uniform Training data tensor format
    dict(type='DavarCollect', keys=['img', 'gt_text'], meta_keys=['img_info']),  # Data content actually involved in training stage
]

print('train_piplines:', train_pipelines)

val_pipeline = [
    dict(type=ppld['File'],
         character=character,
         test_mode=False,
         sensitive=False,
         color_types=["rgb"],    # color loading type, ["rgb", "bgr", "gray"]
         fil_ops=True, ),
    dict(type='ResizeNormalize',
         size=(100, 32),
         interpolation=2,
         mean=img_norm_cfg["mean"],
         std=img_norm_cfg["std"],
         ),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_text'], meta_keys=['img_info']),
]

test_pipeline = [
    dict(type=ppld["File"],
         character=character,
         test_mode=True,
         sensitive=False,
         color_types=["rgb"],
         fil_ops=True, ),
    dict(type='ResizeNormalize',
         size=(100, 32),
         interpolation=0,
         mean=img_norm_cfg["mean"],
         std=img_norm_cfg["std"],
         ),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img'], meta_keys=[]),
]

data = dict(
    samples_per_gpu=192,         # for triplet loss, the samples_per_gpu must can be divided by 3
    workers_per_gpu=2,
    sampler=dict(
        type='MetricSampler',
    ),
    train=dict(
        type=dataset_type,
        batch_ratios=['1.0'],
        dataset=dict(
            type="YORORCGDataset",
            data_type=data_types,
            ann_file=ann_files,
            img_prefix=img_prefixes,
            batch_max_length=25,
            used_ratio=1.0,
            test_mode=False,
            pipeline=train_pipelines)
    ),
    val=dict(
        type=dataset_type,
        batch_ratios=1,
        test_mode=True,
        samples_per_gpu=1920,
        dataset=dict(
            type="YORORCGDataset",
            data_type="File",
            ann_file='/path/to/demo/videotext/datalist/ic13_video_test_datalist.json',
            img_prefix='/path/to/VideoText/IC13/',
            batch_max_length=25,
            used_ratio=0.1,
            test_mode=True,
            pipeline=val_pipeline,)
    ),
    test=dict(
        type=dataset_type,
        batch_ratios=1,
        test_mode=True,
        dataset=dict(
            type="YORORCGDataset",
            data_type='File',
            ann_file='/path/to/demo/videotext/datalist/ic13_video_test_datalist.json',
            img_prefix='/path/to/VideoText/IC13/',
            batch_max_length=25,
            used_ratio=0.1,
            test_mode=True,
            pipeline=test_pipeline, ),
    )
)

"""
3. Training parameter settings
description:
    Configure the corresponding learning rate and related strategy according to the dataset or model structure

Add keywords:
    None

"""
# Optimizer parameter settings
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adadelta', lr=1.0, rho=0.95, eps=1e-8)
# optimizer = dict(type='Adam', amsgrad=False, betas=(0.9, 0.999), eps=1e-8, lr=0.001, weight_decay=0)
optimizer = dict(type='AdamW', betas=(0.9, 0.999), eps=1e-8, lr=0.001, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))


# Learning rate parameter setting
lr_config = dict(
    # policy='fixed',
    policy='step',
    # warmup='linear',
    # warmup_iters=300,
    # warmup_ratio=1.0 / 3,
    gamma=0.3,
    step=[3, 4, 5]
)


# logger setting
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'), ])

"""
======================================================================================================================
                               Evaluation & Checkpoint Instruction manual:
======================================================================================================================
1. Evaluation Setting
----------------------------------------------------------------------------------------------------------------------
  $ General Type
    evaluation = dict(type="DavarDistEvalHook",                # Evaluation Hook Name
                      interval=1,                              # Evaluation interval By Epoch
                      model_type="RECOGNIZOR",                 # Evaluation Model Type, 
                                                               #        including["DETECTOR", "RECOGNIZOR", "SPOTTER"]
                      save_best="accuracy",                    # Save the best metric evaluation model
                      eval_mode="general",                     # Evaluation type, 
                                                               # Note: general equals MMDetection Official 
                                                               #       Evaluation Hook
                      by_epoch=True,                           # by_epoch: True  -- By Epoch
                                                               #           False -- By Iteration
                                                               #                    Note: (Could not work together)
                      rule="greater",                          # the Metric rule, including "greater" or "lower"
                      metric=['accuracy', 'NED'],              # Supported Metric Name
                      )

----------------------------------------------------------------------------------------------------------------------
  $ Lightweight Type
    evaluation = dict(type="DavarDistEvalHook",                # Evaluation Hook Name
                      start=3,                                 # Which epoch to start evaluation
                      start_iter=0.5,                          # the percentage of the training iteration to evaluate
                      save_best="accuracy",                    # Save the best metric evaluation model
                      iter_interval=1,                         # Evaluation interval By Epoch
                      model_type="RECOGNIZOR",                 # Evaluation Model Type, 
                                                               #        including["DETECTOR", "RECOGNIZOR", "SPOTTER"]
                      eval_mode="lightweight",                 # Evaluation type, 
                                                               # Note: lightweight could evaluate the model by 
                                                               #       iterations and by epochs 
                      by_epoch=True,                           # by_epoch: True -- By Epoch
                      by_iter=True,                            # by_iter:  True -- By Iteration 
                                                               #           (independent with By_epoch, 
                                                               #            could work together)
                      rule="greater",                          # the Metric rule, including "greater" or "lower"
                      metric=['accuracy', 'NED'],              # Supported Metric Name
                      )

======================================================================================================================
2. Checkpoint Setting
----------------------------------------------------------------------------------------------------------------------
  $ General Type
    checkpoint_config = dict(type="DavarCheckpointHook",       # Checkpoint Hook Name
                             interval=1,                       # Checkpoint save interval By Epoch
                             by_epoch=True,                    # by_epoch: True  -- By Epoch
                                                               #           False -- By Iteration
                                                               #                    Note: (Could not work together)
                             filename_tmpl='ckpt/ace_e{}.pth', # Checkpoint Save Name format
                             metric="accuracy",                # Save the best metric Name "Accuracy"
                             rule="greater",                   # the Metric rule, including "greater" or "lower"
                             save_mode="general",              # General equals MMDetection Official Checkpoint Hook
                             )

----------------------------------------------------------------------------------------------------------------------
  $ Lightweight Type
    checkpoint_config = dict(type="DavarCheckpointHook",       # Checkpoint Hook Name
                             interval=1,                       # Checkpoint save interval By Epoch
                             iter_interval=1,                  # Checkpoint save interval By Iteration
                             by_epoch=True,                    # by_epoch: True -- By Epoch
                             by_iter=True,                     # by_iter:  True -- By Iteration 
                                                               #           (independent with By_epoch, 
                                                               #            could work together)
                             filename_tmpl='ckpt/ace_e{}.pth', # Checkpoint Save Name format
                             metric="accuracy",                # Save the best metric Name "Accuracy"
                             rule="greater",                   # the Metric rule, including "greater" or "lower"
                             save_mode="lightweight",          # Lightweight type, only save the best metric model and
                                                               # latest iteration and latest epoch model
                             init_metric=-1,                   # initial metric of the model 
                             model_milestone=0.5               # the percentage of the 
                                                               # training process to save checkpoint
                             )
======================================================================================================================
"""

checkpoint_config = dict(type="DavarCheckpointHook",
                         interval=1,
                         iter_interval=1,
                         by_epoch=True,
                         by_iter=False,
                         filename_tmpl='ckpt/res32_att_e{}.pth',
                         metric="accuracy",
                         rule="greater",
                         save_mode="general"
                         # init_metric=-1,
                         # model_milestone=0
                         )

# evaluation setting
evaluation = dict(type="DavarDistEvalHook",
                  start=1,
                  start_iter=None,
                  save_best="accuracy",
                  iter_interval=1,
                  model_type="RECOGNIZOR",
                  eval_mode="lightweight",
                  by_epoch=True,
                  by_iter=False,
                  rule="greater",
                  metric=['accuracy', 'NED'],
                  )

# === runtime settings ===
# yapf:enable
runner = dict(type='EpochBasedRunner', max_epochs=100)                  # Total training epoch
dist_params = dict(backend='nccl')
log_level = 'INFO'

# The path where the model is saved
work_dir = '/path/to/workspace/ic15_att_base/'

# Load from Pre-trained model path
load_from = '/path/to/demo/text_recognition/att/Best_epoch.pth'

# Resume from Pre-trained model path
resume_from = None

# workflow setting
workflow = [('train', 1)]

# gpu number
gpus = 1
