"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_r50_3stages_enlarge_tt.py
# Abstract       :    Model settings for total_text

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31
#########################################################################
"""

_base_= "./tp_det_r50.py"
model = dict(
    backbone=dict(
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 2, 1),   # enlarge receptive field in 8x feature map
        strides=(1, 1, 2)
    ),
    neck=dict(
        in_channels=[256, 512, 1024],
        num_outs=3)
)

data = dict(
    samples_per_gpu=2,
    train=dict(
        ann_file=[
            '/path/to/datalist/total_text_train_datalist.json',
            ],
        img_prefix=[
            '/path/to/Total-Text/',
            ],
        ),
  )

# checkpoint saved path
checkpoint_config = dict(interval=5,  filename_tmpl='checkpoint/tp_r50_3stages_enlarge_tt_epoch_{}.pth')

# 'Pretrained model on Synthtext'
# you can simply load_from = 'path/to/tp_det_r50_tt_e25-45b1f5cf.pth' to fine-tune current model into a new domain
load_from = "/path/to/workspace/log/checkpoint/tp_det_r50_3stages_enlarge_tt-45b1f5cf.pth"
