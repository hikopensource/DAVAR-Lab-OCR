"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_pub.py
# Abstract       :    Script for inference

# Current Version:    1.0.1
# Date           :    2022-05-12
##################################################################################################
"""

import cv2
import mmcv
import json
import jsonlines
import numpy as np
from tqdm import tqdm
from davarocr.davar_table.utils import TEDS, format_html
from davarocr.davar_common.apis import inference_model, init_model


def obtain_ocr_results(img_path, model_textdet, model_rcg):
    """obtrain ocr results of table.
    """

    def crop_from_bboxes(img, bboxes, expand_pixels=(0, 0, 0, 0)):
        """crop images from original images for recognition model
        """
        ret_list = []
        for bbox in bboxes:
            max_x, max_y = min(img.shape[1], bbox[2] + expand_pixels[3]), min(img.shape[0], bbox[3] + expand_pixels[1])
            min_x, min_y = max(0, bbox[0] - expand_pixels[2]), max(0, bbox[1] - expand_pixels[0])
            if len(img.shape) == 2:
                crop_img = img[min_y: max_y, min_x: max_x]
            else:
                crop_img = img[min_y: max_y, min_x: max_x, :]
            ret_list.append(crop_img)

        return ret_list

    ocr_result = {'bboxes': [], 'confidence': [], 'texts': []}

    # single-line text detection
    text_bbox, text_mask = inference_model(model_textdet, img_path)[0]
    text_bbox = text_bbox[0]
    for box_id in range(text_bbox.shape[0]):
        score = text_bbox[box_id, 4]
        box = [int(cord) for cord in text_bbox[box_id, :4]]
        ocr_result['bboxes'].append(box)
        ocr_result['confidence'].append(score)

    # single-line text recognition
    origin_img = mmcv.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    cropped_img = crop_from_bboxes(origin_img, ocr_result['bboxes'], expand_pixels=(1, 1, 3, 3))
    rcg_output = inference_model(model_rcg, cropped_img)
    ocr_result['texts'] = rcg_output['text']

    return ocr_result


# test setting
do_visualize = 0  # whether to visualize
evaluation_structure_only = False  # only evaluate structure or evaluate structure with ocr results
vis_dir = "/path/to/save/visualization"
savepath = "/path/to/save/prediction"

# LGPMA setting
config_lgpma = '/path/to/configs/lgpma_pub.py'
checkpoint_lgpma = 'path/to/lgpma_checkpoint'
model_lgpma = init_model(config_lgpma, checkpoint_lgpma)

# OCR model setting.
config_det = '/path/to/configs/ocr_models/det_mask_rcnn_r50_fpn_pubtabnet.py'
checkpoint_det = '/path/to/text_detection_checkpoint'
config_rcg = '/path/to/configs/ocr_models/rcg_res32_bilstm_attn_pubtabnet_sensitive.py'
checkpoint_rcg = '/path/to/text_recognition_checkpoint'
model_det = init_model(config_det, checkpoint_det)
if 'postprocess' in model_det.cfg['model']['test_cfg']:
    model_det.cfg['model']['test_cfg'].pop('postprocess')
model_rcg = init_model(config_rcg, checkpoint_rcg)

# getting image prefix and test dataset from config file
img_prefix = model_lgpma.cfg["data"]["test"]["img_prefix"]
test_dataset = model_lgpma.cfg["data"]["test"]["ann_file"]
with jsonlines.open(test_dataset, 'r') as fp:
    test_file = list(fp)

# generate prediction of html and save result to savepath
pred_result = dict()
pred_html = dict()
for sample in tqdm(test_file):
    img_path = img_prefix + sample['filename']
    # The ocr results used here can be replaced with your results.
    result_ocr = obtain_ocr_results(img_path, model_det, model_rcg)

    # predict result of table, including bboxes, labels, texts of each cell and html representing the table
    model_lgpma.cfg['model']['test_cfg']['postprocess']['ocr_result'] = [result_ocr]
    result_table = inference_model(model_lgpma, img_path)[0]
    pred_result[sample['filename']] = result_table
    pred_html[sample['filename']] = result_table['html']

    # detection results visualization
    if do_visualize:
        img = cv2.imread(img_path)
        img_name = img_path.split("/")[-1]
        bboxes_non = [b for b in result_table['content_ann']['bboxes'] if len(b)]
        bboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] if len(b) == 4 else b for b in bboxes_non]
        for box in bboxes:
            for j in range(0, len(box), 2):
                cv2.line(img, (box[j], box[j + 1]), (box[(j + 2) % len(box)], box[(j + 3) % len(box)]), (0, 0, 255), 1)
        cv2.imwrite(vis_dir + img_name, img)

with open(savepath, "w", encoding="utf-8") as writer:
    json.dump(pred_result, writer, ensure_ascii=False)

# generate ground-truth of html from pubtabnet annotation of test dataset.
gt_dict = dict()
for data in test_file:
    if data['filename'] not in pred_html.keys():
        continue
    if evaluation_structure_only is False:
        tokens = data['html']['cells']
        for ind, item in enumerate(tokens):
            tok_nofont = [tok for tok in item['tokens'] if len(tok) <= 1]
            tok_valid = [tok for tok in tok_nofont if tok != ' ']
            tokens[ind]['tokens'] = tok_nofont if len(tok_valid) else []
        data['html']['cells'] = tokens
    gt_dict[data['filename']] = {'html': format_html(data)}

# evaluation using script from PubTabNet
teds = TEDS(structure_only=evaluation_structure_only, n_jobs=16)
scores = teds.batch_evaluate(pred_html, gt_dict)
print(np.array(list(scores.values())).mean())
