"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ace_res32.py
# Abstract       :    ACE recognition Model

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""
# encoding=utf-8
_base_ = [
    './baseline.py'
]

# recognition dictionary
character = "/data1/open-source/demo/text_recognition/__dictionary__/Scene_text_68.txt"

"""
1. Model Settings
include model-related setting, such as model type, user-selected modules and parameters.

"""
# model parameters for changing the ace
model = dict(
    sequence_module=dict(
        type='CascadeRNN',
        rnn_modules=[
            dict(
                type='BidirectionalLSTM',
                input_size=512,
                hidden_size=256,
                output_size=512,
                with_linear=True,
                bidirectional=True,), ],
        _delete_=True
    ),
    sequence_head=dict(
        type='ACEHead',
        embed_size=512,
        batch_max_length=25,
        loss_ace=dict(
            type="ACELoss",
            character=character),
        converter=dict(
            type='ACELabelConverter',
            character=character, ),
        _delete_=True
    ),
)

data = dict(
    sampler=dict(
        type='DistBatchBalancedSampler',
        mode=0,
    ),
)

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
                  interval=1,
                  iter_interval=5000,
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
work_dir = '/data1/workdir/davar_opensource/ace/'


# distributed training setting
dist_params = dict(backend='nccl')
