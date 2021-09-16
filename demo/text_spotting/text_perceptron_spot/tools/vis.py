"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    vis.py
# Abstract       :    Script for visualization

# Current Version:    1.0.0
# Date           :    2021-06-24
##################################################################################################
"""
import mmcv
import json
from davarocr.davar_common.apis import inference_model, init_model
import cv2
import time
import json
from show_masks import show_text, show_segmentation, show_cate, show_mask_att

config_file = '../configs/tp_r50_e2e_finetune.py'
checkpoint_file = '../log/checkpoint/tp_r50_e2e_finetune.pth'  # Model weights

model = init_model(config_file, checkpoint_file, device='cuda:0')
cfg = model.cfg

test_dataset = '../../datalist/icdar2013_test_datalist.json'
img_prefix = '/path/to/ICDAR2013-Focused-Scene-Text/'

with open(test_dataset) as load_f:
    test_file = json.load(load_f, encoding="utf-8" )
cnt = 0
time_sum = 0.0
out_dict = {}
for filename in test_file:
    # Load images
    img_path= img_prefix + filename
    img = mmcv.imread(img_path)
    img_copy = img.copy()
    img_name = img_path.split("/")[-1]

    # Inference
    print('predicting {} - {}'.format(cnt, img_path))
    time_start = time.time()
    result = inference_model(model, img)[0]
    time_end = time.time()
    time_sum += (time_end - time_start)
    
    final_text_results = result['texts']
    final_box_results = result['points']
    
    show_text(img, final_text_results, final_box_results, out_prefix="../vis/" + filename.split("/")[-1][:-4])
    cnt += 1
print('FPS: {}'.format(cnt / time_sum))
print('total time: {}'.format(time_sum))



