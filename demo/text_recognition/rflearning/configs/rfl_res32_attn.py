"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    rfl_res32_attn.py
# Abstract       :    RF-learning recognition Model

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""
# encoding=utf-8
_base_ = [
    './baseline.py'
]

# recognition dictionary
character = "/data1/open-source/demo/text_recognition/__dictionary__/Scene_text_36.txt"

"""
1. Model Settings
include model-related setting, such as model type, user-selected modules and parameters.

"""
# model parameters for changing the rf-learning
model = dict(
    type='RFLRecognizor',
    transformation=dict(
        type='TPS_SpatialTransformer',
        F=20,
        I_size=(32, 100),
        I_r_size=(32, 100),
        I_channel_num=1,
    ),
    backbone=dict(
        type='ResNetRFL',
        input_channel=1,
        output_channel=512,),
    # neck_s2v=None,                                # Training strategy
    # neck_v2s=None,                                # Step1: training total RF-Learning, train_type="visual",
    neck_v2s=dict(                              # Step2: training total RF-Learning, train_type="total",
        type='V2SAdaptor',                      # neck_v2s=V2SAdaptor, neck_s2v=S2VAdaptor
        in_channels=512,),
    neck_s2v=dict(
        type='S2VAdaptor',
        in_channels=512,),
    counting_head=dict(
        type='CNTHead',
        embed_size=512,
        encode_length=26,
        loss_count=dict(
            type="MSELoss",
            reduction='mean'),
        converter=dict(
            type='RFLLabelConverter',
            character=character, ),),
    sequence_head=dict(
        type='AttentionHead',
        input_size=512,
        hidden_size=256,
        batch_max_length=25,
        converter=dict(
            type='AttnLabelConverter',
            character=character,
            use_cha_eos=True, ),
        loss_att=dict(
            type='StandardCrossEntropyLoss',
            ignore_index=0,
            reduction='mean',
            loss_weight=1.0),),
    # train_type="visual",
    train_type="total",
    _delete_=True
    # Step1: train_type="visual"
    # Step2: train_type="semantic",
    # Step3: train_type="total"
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
    'LMDB_Standard',
    'LMDB_Standard'
]

# File prefix path of the traning dataset
img_prefixes = [
    '*****/TextRecognition/LMDB/BenchEn/train/',  # path to the training dataset
    '*****/TextRecognition/LMDB/BenchEn/train/',  # path to the training dataset
]


# Dataset Name
ann_files = [
    'MJ', 'SK'
]

# Training dataset load type
dataset_type = 'DavarMultiDataset'

# Normalization parameter
img_norm_cfg = dict(
    mean=[127.5],
    std=[127.5])

# training pipeline parameter
train_pipelines = [
    dict(
        type=ppld["LMDB_Standard"],
        character=character,        # recognition dictionary
        test_mode=False,            # whether is in test mode
        sensitive=False,            # sensitive to Upper or Lower
        color_types=["gray"],       # color loading type, ["rgb", "bgr", "gray"]
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
    dict(type='DavarCollect', keys=['img', 'gt_text']),  # Data content actually involved in training stage
]

print('train_piplines:', train_pipelines)

val_pipeline = [
    dict(type=ppld["LMDB_Standard"], 
         character=character,
         test_mode=True,
         sensitive=False,
         color_types=["gray"],    # color loading type, ["rgb", "bgr", "gray"]
         fil_ops=True, ),
    dict(type='ResizeNormalize',
         size=(100, 32),
         interpolation=2,
         mean=img_norm_cfg["mean"],
         std=img_norm_cfg["std"],
         ),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_text'], meta_keys=[]),
]

test_pipeline = [
    dict(type=ppld["LMDB_Standard"],
         character=character,
         test_mode=True,
         sensitive=False,
         color_types=["gray"],
         fil_ops=True, ),
    dict(type='ResizeNormalize',
         size=(100, 32),
         interpolation=2,
         mean=img_norm_cfg["mean"],
         std=img_norm_cfg["std"],
         ),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img'], meta_keys=[]),
]

data = dict(
    samples_per_gpu=128,         # batchsize=100->memory:6400M
    workers_per_gpu=2,
    sampler=dict(
        type='DistBatchBalancedSampler',  # BatchBalancedSampler or DistBatchBalancedSampler
        mode=0,
        # model 0: Balance in batch, calculate the epoch according to the first iterative data set
        # model 1: Balance in batch, calculate the epoch according to the last iterative data set
        # model 2: Balance in batch, record unused data
        # model -1: Each dataset is directly connected and shuffled
    ),
    train=dict(
        type=dataset_type,
        batch_ratios=['0.5', '0.5'],
        dataset=dict(
            type="DavarRCGDataset",
            data_type=data_types,
            ann_file=ann_files,
            img_prefix=img_prefixes,
            batch_max_length=25,
            used_ratio=1,
            test_mode=False,
            pipeline=train_pipelines)
    ),
    val=dict(
        type=dataset_type,
        batch_ratios=1,
        samples_per_gpu=400,
        test_mode=True,
        dataset=dict(
            type="DavarRCGDataset",
            data_type="LMDB_Standard",
            ann_file='mixture',
            img_prefix='/path/to/validation/',
            batch_max_length=25,
            used_ratio=1,
            test_mode=True,
            pipeline=val_pipeline,)
    ),
    test=dict(
        type=dataset_type,
        batch_ratios=1,
        test_mode=True,
        dataset=dict(
            type="DavarRCGDataset",
            data_type='LMDB_Standard',
            ann_file='IIIT5k_3000',
            img_prefix='/path/to/evaluation/',
            batch_max_length=25,
            used_ratio=1,
            test_mode=True,
            pipeline=test_pipeline, ),
    )
)


"""
4. Runtime Settings
include information about checkpoint, logging, evaluation, workflow, 
pretrained models and other defined parameters during runtime.
"""

checkpoint_config = dict(type="DavarCheckpointHook",
                         interval=1,
                         iter_interval=5000,
                         by_epoch=True,
                         by_iter=True,
                         filename_tmpl='ckpt/res32_ace_e{}.pth',
                         metric="accuracy",
                         rule="greater",
                         save_mode="lightweight",
                         init_metric=-1,
                         model_milestone=0.5
                         )

# logger setting
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), ])


evaluation = dict(start=3,
                  start_iter=0.5,
                  save_best="accuracy",
                  interval=1,
                  iter_interval=5000,
                  model_type="RECOGNIZOR",
                  eval_mode="lightweight",
                  by_epoch=True,
                  by_iter=True,
                  rule="greater",
                  metric=['accuracy', 'NED'],
                  )

# evaluation = dict(type="DavarDistEvalHook",
#                   interval=1,
#                   model_type="recognizor",
#                   save_best="accuracy",
#                   eval_mode="general",
#                   by_epoch=True,
#                   rule="greater",
#                   metric=['accuracy', 'NED'],
#                   )

# runner setting
runner = dict(type='EpochBasedRunner', max_epochs=6)

# must specify this parameter
find_unused_parameters = True

# Load from Pre-trained model path
load_from = '/path/to/davar_opensource/rflearning_visual/RFL_visual_pretrained-2654bc6b.pth'

# work directory
work_dir = '/path/to/davar_opensource/rflearning_total/'

# distributed training setting
dist_params = dict(backend='nccl')
