"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_chargrid.py
# Abstract       :    Script for inference and visualization

# Current Version:    1.0.0
# Date           :    2022-04-08
##################################################################################################
"""

import cv2
import json
import os.path as osp
import time
from tqdm import tqdm
from davarocr.davar_common.apis import inference_model, init_model
from davarocr.davar_common.datasets import build_dataset

# visualization setting
vis_dir = "./vis/publaynet/res" # path to save visualization results

# path setting
save_pred_path = "./vis/publaynet/pred_res.json" # path to save prediction
config_file = './configs/publaynet_chargrid.py' # config path
checkpoint_file = '/path/to/checkpoint_file'# model path

# loading model from config file and pth file
model = init_model(config_file, checkpoint_file)

# getting image prefix and test dataset from config file
img_prefix = model.cfg["data"]["test"]["img_prefix"]
test_dataset = model.cfg["data"]["test"]["ann_file"]
with open(test_dataset, 'r') as fp:
    test_file = json.load(fp, encoding="utf-8")

# generate prediction and save result to savepath
cnt = 0
time_sum = 0.0
pred_dict = dict()

dataset = build_dataset(model.cfg.data.test)
dataset_type = model.cfg.data.test.type
for idx, filename in enumerate(tqdm(test_file)):
    img_info = dataset.data_infos[idx]
    if dataset_type == 'PublaynetDataset':
        img_info = dataset.pre_prepare(img_info)
    ann_info = img_info.get('ann', None)
    ann_info_2 = img_info.get('ann2', None)
    img_dict = dict(img_info=img_info, ann_info=ann_info, ann_info_2=ann_info_2)
    dataset.pre_pipeline(img_dict)
    tic = time.time()
    if dataset_type == 'PublaynetDataset':
        result = inference_model(model, img_dict)[0][0]
    elif dataset_type == 'WildReceiptDataset':
        result = inference_model(model, img_dict)[0]
    else:
        raise NotImplementedError
    # bbox_result = list(zip(*result))[0]
    time_sum += (time.time() - tic)

    # detection results visualization
    img_path = osp.join(img_prefix, filename)
    img = cv2.imread(img_path)
    img_name = img_path.split("/")[-1]
    bboxes = []
    for per_class_bboxes in result:
        for i in range(per_class_bboxes.shape[0]):
            b = list(map(int, per_class_bboxes[i]))
            bboxes.append([b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]])
    for box in bboxes:
        for j in range(0, len(box), 2):
            cv2.line(img, (box[j], box[j + 1]), (box[(j + 2) % len(box)], box[(j + 3) % len(box)]), (0, 0, 255), 1)
    cv2.imwrite(osp.join(vis_dir, img_name), img)
    # Save results to JSON
    pred_dict[filename] = {
        "height": test_file[filename]['height'],
        "width": test_file[filename]['width'],
        "content_ann": {
            "bboxes": bboxes
        }
    }
    cnt += 1

print('FPS: {}'.format(cnt / time_sum))
print('total time: {}'.format(time_sum))

with open(save_pred_path, "w", encoding="utf-8") as writer:
    json.dump(pred_dict, writer, ensure_ascii=False)

