"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    evaluation.py
# Abstract       :    Table understanding evaluation metrics.

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""
import sys
import json
from davarocr.davar_table.core.evaluation import evaluate_tree_f1, evaluate_cellcls_f1

if __name__ == '__main__':
    # load ann data
    jsonpath = '/path/to/gt.json'
    predpath = '/path/to/pred.json'

    if len(sys.argv) == 3:
        jsonpath = sys.argv[1]
        predpath = sys.argv[2]

    with open(jsonpath, 'r', encoding='utf-8') as fp:
        ann_data = json.load(fp)
    with open(predpath, 'r', encoding='utf-8') as fp:
        ann_pred = json.load(fp)
    assert len(ann_data) == len(ann_pred)

    # table items extraction evaluation
    truth_list = []
    preds_list = []
    for imgid, imgname in enumerate(ann_data):
        truth = ann_data[imgname]['content_ann']['relations']
        preds = ann_pred[imgname]['content_ann']['relations']
        truth_list.append(truth)
        preds_list.append(preds)

    hard_recall, hard_precision, hard_f1 = evaluate_tree_f1(preds_list, truth_list, eval_type='hard')
    tree_recall, tree_precision, tree_f1 = evaluate_tree_f1(preds_list, truth_list, eval_type='soft')

    # cell type classification evaluation
    truth_list = []
    preds_list = []
    for imgid, imgname in enumerate(ann_data):
        truth = ann_data[imgname]['content_ann']['labels']
        preds = ann_pred[imgname]['content_ann']['labels']
        truth_list.append(truth)
        preds_list.append(preds)

    tophead, lefthead, data, other = evaluate_cellcls_f1(preds_list, truth_list)
    marco_f1 = (tophead + lefthead + data + other) / 4
    print('tophead={}, lefthead={}, data={}, other={}, marco_f1={}'.format(tophead, lefthead, data, other, marco_f1))
