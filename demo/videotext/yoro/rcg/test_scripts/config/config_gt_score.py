"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_ace.py
# Abstract       :    ACE Model evaluation config

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""

import os

_base_ = [
    '/path/to/demo/videotext/yoro/rcg/configs/ic15_rgb_res32_bilstm_attn.py'
]

# save result of the test experiment
out_path = os.path.join('/path/to/output/')


testsets = [
    {
        'Name': 'IC15_videotext',
        'FilePre': '/path/to/VideoText/IC15/Images',
        'AnnFile': '/path/to/video_ic15_train_datatlist_filterLess3_quality.json',
    },

]


data = dict(
    imgs_per_gpu=400,  # 128
    workers_per_gpu=2,  # 2
    sampler=dict(
        type='BatchBalancedSampler',
        mode=0,),
    test=dict(
        filter_cares=False,)
)

ckpts = list()

# model name setting
out_name = 'davar_test_att_new'

# model parameter dictionary
tmp_dict = dict()

# experiment Name
tmp_dict['Name'] = 'att_new'

# ===================== model .pth file path ========================
tmp_dict['ModelPath'] = 'path/to/ckpt/res32_att_e3-f736fbed.pth'
out_name += '/' + tmp_dict['ModelPath'].split('/')[-2].split('.')[0]

# ===================== model config file path ========================
tmp_dict['ConfigPath'] = '/path/to/demo/videotext/yoro/rcg/configs/ic15_rgb_res32_bilstm_attn.py'

# ===================== model test mode ========================
tmp_dict['Epochs'] = None
ckpts.append(tmp_dict)

force_test = True
force_eval = True


gen_train_score = True
pred_test_score = False


test_path = out_path + 'res/'
eval_path = out_path + 'eval/'

