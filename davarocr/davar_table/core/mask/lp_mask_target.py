"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    lp_mask_target.py
# Abstract       :    Produce local pyramid mask according to gt_bbox and gt_mask.

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

from math import ceil
import numpy as np
from .structures import BitmapMasksTable


def get_lpmasks(gt_masks, gt_bboxes):
    """Produce local pyramid mask according to gt_bbox and gt_mask (for a batch of imags).

    Args:
        gt_masks(list(BitmapMasks)): masks of the text regions
        gt_bboxes(list(Tensor)): bboxes of the aligned cells

    Returns:
        list(BitmapMasks):pyramid masks in horizontal direction
        list(BitmapMasks):pyramid masks in vertical direction
    """

    gt_masks_temp = map(get_lpmask_single, gt_masks, gt_bboxes)
    gt_masks_temp = list(gt_masks_temp)
    gt_lpmasks_hor = [temp[0] for temp in gt_masks_temp]
    gt_lpmasks_ver = [temp[1] for temp in gt_masks_temp]

    return gt_lpmasks_hor, gt_lpmasks_ver


def get_lpmask_single(gt_mask, gt_bbox):
    """Produce local pyramid mask according to gt_bbox and gt_mask ((for one image).

    Args;
        gt_mask(BitmapMasks): masks of the text regions (for one image)
        gt_bbox(Tensor): (n x 4).bboxes of the aligned cells (for one image)

    Returns;
        BitmapMasksTable;pyramid masks in horizontal direction (for one image)
        BitmapMasksTable;pyramid masks in vertical direction (for one image)
    """

    (num, high, width) = gt_mask.masks.shape
    mask_s1 = np.zeros((num, high, width), np.float32)
    mask_s2 = np.zeros((num, high, width), np.float32)
    for ind, box_text in zip(range(num), gt_mask.masks):
        left_col, left_row, right_col, right_row = list(map(float, gt_bbox[ind, 0:4]))
        x_min, y_min, x_max, y_max = ceil(left_col), ceil(left_row), ceil(right_col) - 1, ceil(right_row) - 1
        middle_x, middle_y = round(np.where(box_text == 1)[1].mean()), round(np.where(box_text == 1)[0].mean())

        # Calculate the pyramid mask in horizontal direction
        col_np = np.arange(x_min, x_max + 1).reshape(1, -1)
        col_np_1 = (col_np[:, :middle_x - x_min] - left_col) / (middle_x - left_col)
        col_np_2 = (right_col - col_np[:, middle_x - x_min:]) / (right_col - middle_x)
        col_np = np.concatenate((col_np_1, col_np_2), axis=1)
        mask_s1[ind, y_min:y_max + 1, x_min:x_max + 1] = col_np

        # Calculate the pyramid mask in vertical direction
        row_np = np.arange(y_min, y_max + 1).reshape(-1, 1)
        row_np_1 = (row_np[:middle_y - y_min, :] - left_row) / (middle_y - left_row)
        row_np_2 = (right_row - row_np[middle_y - y_min:, :]) / (right_row - middle_y)
        row_np = np.concatenate((row_np_1, row_np_2), axis=0)
        mask_s2[ind, y_min:y_max + 1, x_min:x_max + 1] = row_np

    mask_s1 = BitmapMasksTable(mask_s1, high, width)
    mask_s2 = BitmapMasksTable(mask_s2, high, width)

    return mask_s1, mask_s2
