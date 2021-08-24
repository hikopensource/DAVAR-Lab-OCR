"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_track_config.py
# Abstract       :    generate track sequence config

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""

import os

_base_ = [
    '/path/to/demo/text_recognition/__base__/test_base_setting.py'
]

# recognition dictionary
character = "/path/to/demo/text_recognition/__dictionary__/Scene_text_36.txt"

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])

ppld = {
    'LMDB_Standard': 'LoadImageFromLMDB',  # open-source LMDB data
    # Davar dataset type
    'LMDB_Davar': 'RCGLoadImageFromLMDB',
    'File': 'RCGLoadImageFromFile',
    'Loose': 'RCGLoadImageFromLoose',
    'Tight': 'RCGLoadImageFromTight',
}

test_pipeline = [
    dict(type=ppld["File"],
         character=character,
         test_mode=True,
         sensitive=False,
         color_types=["rgb"],
         fil_ops=False, ),
    dict(type='ResizeNormalize',
         size=(100, 32),
         interpolation=0,
         mean=img_norm_cfg["mean"],
         std=img_norm_cfg["std"],
         ),
    dict(type='DavarDefaultFormatBundle'),
    dict(type='DavarCollect', keys=['img'], meta_keys=['img_info']),
]


# The detection result as input
testsets = [
    {
        'Name': 'IC15_spotting',
        'FilePre': '/path/to/Images/',
        'AnnFile': '/path/to/det/ic15_predict_det.json',
        'Type': 'File',
        'PipeLine': test_pipeline,
    },
]

ckpts = list()


# model parameter dictionary
tmp_dict = dict()

# ===================== track hyper parameters ========================
# track output save file
out_dir = '/path/to/save/'
out_file = 'IC15_pred_track_result.json'

feat_sim_thresh = 0.9
feat_sim_with_loc_thresh = 0.85
max_exist_duration = 8
feat_channels = 256
eps = 1e-7

# ===================== merge hyper parameters ========================
# merge output save file
merge_out_dir = '/path/to/save/'
merge_out_file = 'IC15_pred_track_result_merge.json'
merge_max_interval = 10
merge_thresh_tight = 0.45
merge_thresh_loose = 0.2

edit_dist_iou_thresh_tight = 0.4
edit_dist_iou_thresh_loose = 0.3

frame_min_index = 1e15
frame_max_index = -1
max_constant = 1e15



# ===================== model .pth file path ========================
tmp_dict['ModelPath'] = '/path/to/davar_opensource/model/ckpt/IC15_qscore-0dcdad8d.pth'


# ===================== model config file path ========================
tmp_dict['ConfigPath'] = '/path/to/demo/videotext/yoro/rcg/att/configs/ic15_qscore_rgb_res32_bilstm_attn.py'

# ===================== model test mode ========================
ckpts.append(tmp_dict)

