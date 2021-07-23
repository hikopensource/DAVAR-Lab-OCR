"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    east_r50_rbox.py
# Abstract       :    Model settings for EAST detector on ICDAR2015(RBOX mode)

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""
# model_setting
_base_ = './east_r50_rbox.py'
model = dict(
    mask_head=dict(
        loss_reg=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1, reduction='mean'),  # Support 'QUAD'
        geometry='QUAD'),
    test_cfg=dict(
        postprocess=dict(
            nms_method='QUAD'
        )
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='DavarLoadAnnotations', with_poly_bbox=True, with_care=True),
    dict(type='RandomRotate', angles=(-15, 15), borderValue=(0, 0, 0)),
    dict(type='DavarResize', img_scale=[(768, 512), (1920, 1080)], multiscale_mode='range', keep_ratio=True),
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='EASTDataGeneration', geometry='QUAD'),
    dict(type='SegFormatBundle'),
    dict(type='DavarCollect', keys=['img', 'gt_masks']),
]

f_name = 'SR_east_res50_ic15_quad'
checkpoint_config = dict(interval=20, filename_tmpl='checkpoint/'+f_name+'_epoch{}.pth')
