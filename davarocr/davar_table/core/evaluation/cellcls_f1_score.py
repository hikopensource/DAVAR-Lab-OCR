"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    cellcls_f1_score.py
# Abstract       :    Cell type classification evaluation metrics.

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""
import numpy as np


def evaluate_cellcls_f1(preds, gts, class_num=4):
    """Evaluate the prediction with similarity.

    Args:
        preds (list(list(int): The predicted cell types of all samples.
        gts (list(list(int): The ground truth cell types of all samples.
        class_num (int): number of cell type such as left header, top header, data and other.

    Returns:
        list(float): average f1-score (each table has the same weight) of each type of cells.
    """
    cellcls_f1_pertab = [[] for _ in range(class_num)]
    for pred, gt in zip(preds, gts):
        pred_np = np.array([v[0] for v in pred])
        gt_np = np.array([v[0] for v in gt])
        for lab_id in range(class_num):  # th, lh, data, other
            preds_num = (pred_np == lab_id).sum()
            gts_num = (gt_np == lab_id).sum()
            if (not gts_num) and (not preds_num):
                continue
            correct_num = ((pred_np == lab_id) & (gt_np == lab_id)).sum()
            f1_id = 2 * correct_num / (preds_num + gts_num)
            cellcls_f1_pertab[lab_id].append(f1_id)

    cellcls_f1_avg = [sum(cellcls_f1_pertab[lab_id]) / len(cellcls_f1_pertab[lab_id]) for lab_id in range(class_num)]

    return cellcls_f1_avg
