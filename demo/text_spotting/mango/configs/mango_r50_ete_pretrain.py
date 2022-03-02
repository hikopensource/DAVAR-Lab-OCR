"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango_r50_ete_pretrain.py
# Abstract       :    Model settings for mango end-to-end pretrain on synthdata.

# Current Version:    1.0.0
# Date           :    2020-06-24
######################################################################################################
"""
_base_ = './__base__.py'

model = dict(
    character_mask_att_head=dict(
        loss_char_mask_att=None
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=[
            '/path/to/datalist/synthtext_80w.json',              # SynthText 80W
            '/path/to/datalist/train_syntext_word_eng.json',     # SynthText_Curve Part1
            '/path/to/datalist/train_emcs_imgs.json',            # SynthText_Curve Part2
        ],
        img_prefix=[
            '/path/to/SynthText/',
            '/path/to/Syntext_curve/',
            '/path/to/Syntext_curve/',
        ]
    ),
    val=dict(
        ann_file='/path/to/datalist/icdar2013_test_datalist.json',
        img_prefix='/path/to/ICDAR2013-Focused-Scene-Text/',
    ),
    test=dict(
        ann_file='/path/to/datalist/icdar2013_test_datalist.json',
        img_prefix='/path/to/ICDAR2013-Focused-Scene-Text/',
    )
)
optimizer=dict(lr=1e-3)
lr_config = dict(step=[4, 7, 9])
runner = dict(max_epochs=10)
checkpoint_config = dict(interval=1, filename_tmpl='checkpoint/res50_ete_pretrain_epoch_{}.pth')
work_dir = '/path/to/workspace/log/'
load_from = '/path/to/res50_att_pretrain.pth'
