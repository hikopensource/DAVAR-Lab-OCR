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
# recognition dictionary

# Model setting
model = dict(
    sequence_head=dict(          # Recognition head parameter
        loss_triplet=dict(
            type='TripletLoss',
            margin=1.0,
            reduction='mean')
        )
)

# Model training and test parameter configuration
train_cfg = dict(
    fix_rcg=True,
    fix_track=False,
    fix_qscore=True
)

checkpoint_config = dict(interval=10)

# evaluation setting
evaluation = dict(
                  start=10,
                  iter_interval=10
                  )

# === runtime settings ===

# The path where the model is saved
work_dir = '/path/to/davar_opensource/ckpt/save_track/'

# Load from Pre-trained model path
load_from = '/path/to/davar_opensource/ckpt/res32_att_e3-f736fbed.pth'
