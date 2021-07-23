"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_att.py
# Abstract       :    Attn Model evaluation config

# Current Version:    1.0.0
# Date           :    2021-06-11
##################################################################################################
"""

import os

_base_ = [
    '../test_base_setting.py'
]

ckpts = list()

# model name setting
out_name = 'davar_test_att'

# model parameter dictionary
tmp_dict = dict()

# experiment Name
tmp_dict['Name'] = 'davar_test_att'

# ===================== model .pth file path ========================
tmp_dict['ModelPath'] = '/data1/workdir/davar_opensource/att_test/Attn_pretrained-4d81ec6a.pth'
out_name += '/' + tmp_dict['ModelPath'].split('/')[-2].split('.')[0]

# ===================== model config file path ========================
tmp_dict['ConfigPath'] = '/data1/open-source/demo/text_recognition/__base__/res32_bilstm_attn.py'

# ===================== model test mode ========================
tmp_dict['Epochs'] = None
ckpts.append(tmp_dict)


# save result of the test experiment
out_path = os.path.join('/data1/output_dir/sota_exp', out_name + '/')

force_test = False
force_eval = False

do_test = 1  # 1 for test
do_eval = 1

test_path = out_path + 'res/'
eval_path = out_path + 'eval/'

