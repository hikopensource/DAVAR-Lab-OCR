"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    lgpma_pub.py
# Abstract       :    Model settings for LGPMA detector on PubTabNet

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

_base_ = "./lgpma_base.py"

data_fp = f"/home/julian/data/pubtabnet/pubtabnet/"
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=1,
    train=dict(
        ann_file='path/to/PubTabNet_datalist_train_detection.json',
        img_prefix='path/to/PubTabNet'),
    val=dict(
        ann_file=f'{data_fp}PubTabNet_2.0.0_val.jsonl',
        img_prefix=f'{data_fp}/val/'),
    test=dict(
        samples_per_gpu=1,
        ann_file=f'{data_fp}PubTabNet_2.0.0_val.jsonl',
        img_prefix=f'{data_fp}/val/')
)

# yapf:enable
# runtime settings

checkpoint_config = dict(interval=1, filename_tmpl='checkpoint/maskrcnn-lgpma-pub-e{}.pth')

work_dir = 'path/to/workdir'
