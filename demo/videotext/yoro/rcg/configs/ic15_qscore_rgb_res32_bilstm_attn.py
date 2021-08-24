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
    './ic15_rgb_res32_bilstm_attn.py'
]

# Model setting
model = dict(
    sequence_head=dict(          # Head parameter
        loss_l1=dict(
            type='L1Loss',
            reduction='mean'
        ),
    )
)

# Model training and test parameter configuration
train_cfg = dict(                # Dimensions remain or change
    fix_rcg=True,
    fix_track=True,
    fix_qscore=False
)




# File prefix path of the traning dataset
img_prefixes = [
    '/path/to/IC15/Images/',
]

# Dataset Name
ann_files = [
    '/path/to/video_ic15_train_score.json'
]


data = dict(
    train=dict(
        dataset=dict(
            ann_file=ann_files,
            img_prefix=img_prefixes,
            filter_scores=True,)
    ),
)

"""
3. Training parameter settings
description:
    Configure the corresponding learning rate and related strategy according to the dataset or model structure

Add keywords:
    None

"""
# Optimizer parameter settings
optimizer = dict(type='AdamW', betas=(0.9, 0.999), eps=1e-8, lr=0.0003, weight_decay=0)

optimizer_config= dict()


# evaluation setting
evaluation = dict(
                  start=5,
                  iter_interval=5,
                  )

# === runtime settings ===
# yapf:enable
runner = dict(type='EpochBasedRunner', max_epochs=20)

# The path where the model is saved
work_dir = '/path/to/davar_opensource/workdir/'

# Load from Pre-trained model path
load_from = '/path/to/davar_opensource/ckpt/res32_att_e3-f736fbed.pth'

# Resume from Pre-trained model path
resume_from = None

