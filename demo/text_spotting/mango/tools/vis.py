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

config_file = '../configs/mango_r50_ete_finetune.py'
checkpoint_file = '../log/checkpoint/res50_ete_finetune_ic13.pth'  # Model weights

cfg_options = dict(model=dict(test_cfg=dict(postprocess=dict(do_visualization=True))))

model = init_model(config_file, checkpoint_file, cfg_options=cfg_options, device='cuda:0')
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
    cate_preds = result['cate_preds']
    seg_preds = result['seg_preds']
    mask_att_preds = result['character_mask_att_preds']

    # Visualization Segmentation and Mask Attention
    show_segmentation(img, seg_preds, final_box_results, out_prefix="../vis/" + filename.split("/")[-1][:-4])
    
    resize_shape = cfg.data.test.pipeline[1]['img_scale']
    
    show_cate(img, cate_preds, resize_shape=resize_shape, pad_size_divisor=128, out_prefix="../vis/" + filename.split("/")[-1][:-4])
    
    if mask_att_preds is not None:
        show_mask_att(img, mask_att_preds,out_prefix="../vis/" + filename.split("/")[-1][:-4])
    
    show_text(img, final_text_results, final_box_results, out_prefix="../vis/" + filename.split("/")[-1][:-4])
    cnt += 1
print('FPS: {}'.format(cnt / time_sum))
print('total time: {}'.format(time_sum))



