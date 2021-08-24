"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_base_setting.py
# Abstract       :    Base recognition Model test setting

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""

# encoding=utf-8

# recognition dictionary
character = "/path/to/demo/text_recognition/__dictionary__/Scene_text_68.txt"

# dataset settings
dataset_type = 'DavarMultiDataset'

img_norm_cfg = dict(
    mean=[127.5], std=[127.5])


ppld = {
    'LMDB_Standard': 'LoadImageFromLMDB',  # open-source LMDB data

    # Davar dataset type
    'LMDB_Davar': 'RCGLoadImageFromLMDB',
    'File': 'RCGLoadImageFromFile',
    'Loose': 'RCGLoadImageFromLoose',
    'Tight': 'RCGLoadImageFromTight',
}

test_pipeline = [
    dict(type='LoadImageFromLMDB',
         character=character,
         sensitive=False,
         color_types=['gray'],
         fil_ops=True),
    dict(type='ResizeNormalize',
         size=(100, 32),
         interpolation=2,
         mean=[127.5],
         std=[127.5]),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect',
         keys=['img'],
         meta_keys=[])]

testsets = [
    {
        'Name': 'IIIT5k',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'IIIT5k_3000/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'SVT',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'SVT/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'IC03_860',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'IC03_860/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'IC03_867',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'IC03_867/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'IC13_857',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'IC13_857/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'IC13_1015',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'IC13_1015/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'IC15_1811',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'IC15_1811/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'IC15_2077',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'IC15_2077/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'SVTP',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'SVTP/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },
    {
        'Name': 'CUTE80',
        'FilePre': '/path/to/evaluation/',
        'AnnFile': 'CUTE80/',
        'Type': 'LMDB_Standard',
        'PipeLine': test_pipeline,
    },

]

# data setting
data = dict(
    imgs_per_gpu=400,  # 128
    workers_per_gpu=2,  # 2
    sampler=dict(
        type='BatchBalancedSampler',
        mode=0,),
    train=None,
    test=dict(
        type="DavarRCGDataset",
        info=testsets,
        batch_max_length=25,
        used_ratio=1,
        test_mode=True,
        pipeline=test_pipeline)

)

# runtime setting
dist_params = dict(backend='nccl')
launcher = 'none'
