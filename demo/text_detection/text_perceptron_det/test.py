"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test.py
# Abstract       :    Script for inference and visualization

# Current Version:    1.0.0
# Date           :    2020-05-31
#########################################################################
"""

import mmcv
from davarocr.davar_common.apis import inference_model, init_model
import cv2
import time
import json
import os

# =============== Settings for Total-Text ===================================
# config_file = './config/tp_det_r50_3stages_enlarge_tt.py'
# checkpoint_file = './log/checkpoint/tp_det_r50_3stages_enlarge_tt-45b1f5cf.pth'
# test_dataset= '../datalist/total_datalist.json'
# img_prefix = '/path/to//Total-Text/'

# =============== Settings for SCUT-CTW1500 ===================================
config_file = './config_det/tp_det_r50_3stages_enlarge_ctw.py'
checkpoint_file = './log/checkpoint/tp_det_r50_3stages_enlarge_ctw-c1bf44e7.pth'
test_dataset= '../datalist/ctw1500_test_datalist.json'
img_prefix = '/path/to/SCUT-CTW1500/'


out_put_dir = "./score/"         # path to save final prediction in .txt format
if not os.path.exists(out_put_dir):
    os.mkdir(out_put_dir)
vis_dir = "./vis/"               # path to save visualization result.
if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)
model = init_model(config_file, checkpoint_file, device='cuda:0')

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
    result = inference_model(model, img_path)
    time_end = time.time()
    time_sum += (time_end - time_start)
    print(result)

    # Save pred in txt format
    txt = open(out_put_dir+"{}.txt".format(filename.split("/")[-1].split(".")[0]), "w")
    bboxes = []
    for i in range(len(result["points"])):
        points2 = result["points"][i]
        for j in range(0, len(points2), 2):
            cv2.circle(img_copy, (points2[j], points2[(j+1)]), 5, (0, 255, 255),-1)
            cv2.line(img_copy, (points2[j], points2[j+1]), (points2[(j+2)%len(points2)], points2[(j+3)%len(points2)]),
                     (0, 0, 255), 2)
            txt.write("{},{}".format(points2[j], points2[j+1]))
            if j != len(points2)-2:
                txt.write(",")
            elif i != len(result["points"])-1:
                txt.write("\n")

        points = list(map(int, points2))
        bboxes.append(points)
    txt.close()

    # Save results to JSON
    out_dict[filename]={
        "height":test_file[filename]["height"],
        "width":test_file[filename]["width"],
        "content_ann":{
            "bboxes":bboxes
        }

    }
    # Results visualization
    cv2.imwrite(vis_dir + img_name, img_copy)
    cnt += 1
print('FPS: {}'.format(cnt / time_sum))
print('total time: {}'.format(time_sum))

with open("total_pred.json", "w") as write_f:
    json.dump(out_dict, write_f, ensure_ascii=False, indent=4)

