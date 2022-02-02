"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_pub.py
# Abstract       :    Script for inference

# Current Version:    1.0.1
# Date           :    2021-09-23
##################################################################################################
"""

import cv2
import json
import jsonlines
import numpy as np
from tqdm import tqdm
from eval_pub.metric import TEDS
from eval_pub.format import format_html
from davarocr.davar_common.apis import inference_model, init_model

# visualization setting
do_visualize = 1  # whether to visualize

tsr_fp = f"/home/julian/code/acx-parser-engine/projects/acx-tsr/"
davar_fp = f"{tsr_fp}DAVAR-Lab-OCR/"
lgpma_fp = f"{davar_fp}demo/table_recognition/lgpma/"
results_fp = f"{tsr_fp}results/"
vis_dir = f"{results_fp}visual"  # path to save visualization results

# path setting
savepath = f"{results_fp}res.json"  # path to save prediction
config_file = f"{lgpma_fp}/configs/lgpma_pub.py"  # config path
checkpoint_file = f"{tsr_fp}models/maskrcnn-lgpma-pub-e12-pub.pth"  # model path

# loading model from config file and pth file
model = init_model(config_file, checkpoint_file, device="cpu")

# getting image prefix and test dataset from config file
img_prefix = model.cfg["data"]["test"]["img_prefix"]
test_dataset = model.cfg["data"]["test"]["ann_file"]
with jsonlines.open(test_dataset, "r") as fp:
    test_file = list(fp)

# generate prediction of html and save result to savepath
pred_dict = dict()
for idx, sample in enumerate(tqdm(test_file)):
    # predict html of table
    img_path = img_prefix + sample["filename"]
    result = inference_model(model, img_path)[0]
    pred_dict[sample["filename"]] = result["html"]

    # detection results visualization
    if do_visualize:
        img = cv2.imread(img_path)
        img_name = img_path.split("/")[-1]
        bboxes = [
            [b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in result["bboxes"]
        ]
        for box in bboxes:
            for j in range(0, len(box), 2):
                cv2.line(
                    img,
                    (box[j], box[j + 1]),
                    (box[(j + 2) % len(box)], box[(j + 3) % len(box)]),
                    (0, 0, 255),
                    1,
                )
        cv2.imwrite(vis_dir + img_name, img)

with open(savepath, "w", encoding="utf-8") as writer:
    json.dump(pred_dict, writer, ensure_ascii=False)

# generate ground-truth of html from pubtabnet annotation of test dataset.
gt_dict = dict()
for data in test_file:
    if data["filename"] in pred_dict.keys():
        str_true = data["html"]["structure"]["tokens"]
        gt_dict[data["filename"]] = {"html": format_html(data)}

# evaluation using script from PubTabNet
teds = TEDS(structure_only=True, n_jobs=16)
scores = teds.batch_evaluate(pred_dict, gt_dict)
print(np.array(list(scores.values())).mean())
