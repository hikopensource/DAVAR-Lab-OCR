"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    spin_res32_attn.py
# Abstract       :    SPIN transformation recognition Model

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""
# encoding=utf-8

_base_ = [
    './baseline.py'
]


"""
1. Model Settings
include model-related setting, such as model type, user-selected modules and parameters.
"""
# model parameters for changing the SPIN transformation
model = dict(
    transformation=dict(
         type='SPIN_ColorTransformer',
         input_channel=1,
         offsets=True,
         stn=dict(
            # type='TPS_SpatialTransformer',
            # F=20,
            # I_size=(32, 100),
            # I_r_size=(32, 100),
            # I_channel_num=1
            type='Affine_SpatialTransformer',
            I_size=(32, 100),
            I_r_size=(32, 100),
            I_channel_num=1
          ),
         _delete_=True,
    ),
)

data = dict(
    samples_per_gpu=64)

# checkpoint setting
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


# evaluation setting
evaluation = dict(start=3,
                  start_iter=0.5,
                  save_best="accuracy",
                  iter_interval=5000,
                  interval=1,
                  model_type="RECOGNIZOR",
                  eval_mode="lightweight",
                  by_epoch=True,
                  by_iter=True,
                  rule="greater",
                  metric=['accuracy', 'NED'],
                  )

# runner setting
runner = dict(type='EpochBasedRunner', max_epochs=6)


# work directory
work_dir = '/data1/workdir/davar_opensource/spin/'


# distributed training setting
dist_params = dict(backend='nccl')
