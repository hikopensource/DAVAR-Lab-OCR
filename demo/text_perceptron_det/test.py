"""
#########################################################################
# Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test.py
# Abstract       :    The test demo script

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31
#########################################################################
"""

import mmcv
from mmdet.apis import show_result, init_detector, inference_detector
from mmdet.apis import inference_detector
import cv2
import time
import json

config_file = './config/tp_r50_3stages_enlarge.py'
# Model weights for Total-Text
checkpoint_file = './log/checkpoint/tp_det_r50_tt_e25-45b1f5cf.pth'
# Model weights for SCUT-CTW1500
# checkpoint_file = './log/checkpoint/tp_det_r50_ctw-c1bf44e7.pth'

# Init model
model = init_detector(config_file, checkpoint_file, device='cuda:0')
cfg = model.cfg

# Identify the test data list
# Dataslist of Total-Text
test_dataset= './datalist/total_text_test_datalist.json'
img_prefix = '/path/to/Img_prefix/Total-Text/'

# Datalist of SCUT-CTW1500
#test_dataset= './datalist/ctw1500_test_datalist.json'
#img_prefix = '/path/to/Img_prefix/CTW1500/'

out_dir= 'result'

test_file = mmcv.load(test_dataset)
cnt = 0
time_sum = 0.0
out_dict = {}

# Inference and visualize image one by one
for filename in test_file:
    # Load images
    img_path= img_prefix + filename
    img = mmcv.imread(img_path)
    img_copy = img.copy()
    img_name = img_path.split("/")[-1]

    # Inference
    print('predicting {} - {}'.format(cnt, img_path))
    time_start = time.time()
    result = inference_detector(model, img_path)
    time_end = time.time()
    time_sum += (time_end - time_start)
    print(result)

    # Results visualization
    bboxes = []
    for i in range(len(result["points"])):
        points2 = result["points"][i]
        for j in range(0, len(points2), 2):
            cv2.circle(img_copy, (points2[j], points2[(j + 1)]), 5,
                       (0, 255, 255), -1)
            cv2.line(img_copy, (points2[j], points2[j + 1]), (
            points2[(j + 2) % len(points2)], points2[(j + 3) % len(points2)]),
                     (0, 0, 255), 2)
        points = list(map(int, points2))
        bboxes.append(points)

    # Save results to JSON
    out_dict[filename.split("/")[-1]]={
        "height": test_file[filename]["height"],
        "width": test_file[filename]["width"],
        "bboxes": bboxes
    }
    cv2.imwrite("./result/pred_" + img_name, img_copy)
    cnt += 1
print('FPS: {}'.format(cnt / time_sum))
print('total time: {}'.format(time_sum))

with open("total_pred.json","w") as write_f:
    json.dump(out_dict, write_f, ensure_ascii=False, indent=4)

