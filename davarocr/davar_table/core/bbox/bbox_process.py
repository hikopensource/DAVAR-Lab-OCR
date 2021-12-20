"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    bbox_process.py
# Abstract       :    Implementation of bboxes process used in LGPMA.

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

import numpy as np


def recon_noncell(bboxlist, celllist, imgshape, padding=1):
    """ Produce pseudo-bboxes for empty cells

    Args:
        bboxlist (list): (n x 4).Bboxes of text region in each cell(empty cell is noted as [])
        celllist (list): (n x 4).Start row, start column, end row and end column of each cell
        imgshape (tuple): (height, width).The height and width of input image.
        padding (int): If cells in the first/last row/col are all empty, extend them to padding pixels from boundary.

    Returns:
        list(list): (n x 4).Bboxes of text region in each cell (including empty cells)
    """

    cells_non = np.array([b for a, b in zip(bboxlist, celllist) if a])
    bboxes_non = np.array([b for b in bboxlist if b])
    bboxlist_append = bboxlist.copy()
    cellnp = np.array(celllist, dtype='int32')
    for i, bbox in enumerate(bboxlist_append):
        if bbox:
            continue
        row = [cellnp[i, 0], cellnp[i, 2]]
        col = [cellnp[i, 1], cellnp[i, 3]]
        rowindex_top = np.where((cells_non[:, 0] == row[0]))[0]
        rowindex_down = np.where((cells_non[:, 2] == row[1]))[0]
        colindex_left = np.where((cells_non[:, 1] == col[0]))[0]
        colindex_right = np.where((cells_non[:, 3] == col[1]))[0]

        # At least one cell in this row is non-empty.
        if len(rowindex_top):
            ymin = bboxes_non[rowindex_top, 1].min()

        # All cells in this row are empty and this row is the first row.
        elif not row[0]:
            ymin = padding

        # All cells in this row are empty and this row is not the first row.
        else:
            rowindex_top_mod = np.where((cells_non[:, 2] == row[0] - 1))[0]
            span_number = 1
            while len(rowindex_top_mod) == 0 and (row[0] - span_number) > 0:
                span_number += 1
                rowindex_top_mod = np.where((cells_non[:, 2] == row[0] - span_number))[0]
            if len(rowindex_top_mod) == 0:
                ymin = padding
            else:
                ymin = bboxes_non[rowindex_top_mod, 3].max() + padding

        # At least one cell in this row is non-empty.
        if len(rowindex_down):
            ymax = bboxes_non[rowindex_down, 3].max()

        # All cells in this row are empty and this row is the last row.
        elif row[1] >= cells_non[:, 2].max():
            ymax = imgshape[0] - padding

        # All cells in this row are empty and this row is not the last row.
        else:
            rowindex_down_next = np.where((cells_non[:, 0] == row[1] + 1))[0]
            span_number = 1
            while len(rowindex_down_next) == 0 and (row[1] + span_number) <= cells_non[:, 2].max() - 1:
                span_number += 1
                rowindex_down_next = np.where((cells_non[:, 0] == row[1] + span_number))[0]
            if len(rowindex_down_next) == 0:
                ymax = imgshape[0] - padding
            else:
                ymax = bboxes_non[rowindex_down_next, 1].min() - padding

        # At least one cell in this column is non-empty.
        if len(colindex_left):
            xmin = bboxes_non[colindex_left, 0].min()

        # All cells in this column are empty and this column is the first column.
        elif not col[0]:
            xmin = padding

        # All cells in this column are empty and this column is not the last column.
        else:
            colindex1_left_mod = np.where((cells_non[:, 3] == col[0] - 1))[0]
            span_number = 1
            while len(colindex1_left_mod) == 0 and (col[0] - span_number) > 0:
                span_number += 1
                colindex1_left_mod = np.where((cells_non[:, 3] == col[0] - span_number))[0]
            if len(colindex1_left_mod) == 0:
                xmin = padding
            else:
                xmin = bboxes_non[colindex1_left_mod, 2].max() + padding

        # At least one cell in this column is non-empty.
        if len(colindex_right):
            xmax = bboxes_non[colindex_right, 2].max()

        # All cells in this column are empty and this column is the last column.
        elif col[1] > cells_non[:, 3].max():
            xmax = imgshape[1] - padding

        # All cells in this column are empty and this column is not the last column.
        else:
            colindex_right_mod = np.where((cells_non[:, 1] == col[1] + 1))[0]
            span_number = 1
            while len(colindex_right_mod) == 0 and (col[1] + span_number) <= cells_non[:, 3].max() - 1:
                span_number += 1
                colindex_right_mod = np.where((cells_non[:, 1] == col[1] + span_number))[0]
            if len(colindex_right_mod) == 0:
                xmax = imgshape[1] - padding
            else:
                xmax = bboxes_non[colindex_right_mod, 0].min() - padding
        bboxlist_append[i] = list(map(int, [xmin, ymin, xmax, ymax]))

    return bboxlist_append


def recon_largecell(bboxlist, celllist):
    """ Produce pseudo-bboxes for aligned cells

    Args:
        bboxlist (list): (n x 4).Bboxes of text region in each cell (including empty cells)
        celllist (list): (n x 4).Start row, start column, end row and end column of each cell

    Returns:
        list(list): (n x 4).Bboxes of aligned cells (including empty cells)
    """

    bboxlist_align = bboxlist.copy()
    bboxnp = np.array(bboxlist, dtype='int32')
    cellnp = np.array(celllist, dtype='int32')
    for i in range(len(bboxlist)):
        row = [cellnp[i, 0], cellnp[i, 2]]
        col = [cellnp[i, 1], cellnp[i, 3]]
        rowindex1 = np.where((cellnp[:, 0] == row[0]))[0]
        rowindex2 = np.where((cellnp[:, 2] == row[1]))[0]
        colindex1 = np.where((cellnp[:, 1] == col[0]))[0]
        colindex2 = np.where((cellnp[:, 3] == col[1]))[0]
        newbbox = [bboxnp[colindex1, 0].min(), bboxnp[rowindex1, 1].min(), bboxnp[colindex2, 2].max(),
                   bboxnp[rowindex2, 3].max()]
        bboxlist_align[i] = list(map(int, newbbox))

    return bboxlist_align


def rect_max_iou(box_1, box_2):
    """Calculate the maximum IoU between two boxes: the intersect area / the area of the smaller box

    Args:
        box_1 (np.array | list): [x1, y1, x2, y2]
        box_2 (np.array | list): [x1, y1, x2, y2]

    Returns:
        float: maximum IoU between the two boxes
    """

    addone = 0  # 0 in mmdet2.0 / 1 in mmdet 1.0
    box_1, box_2 = np.array(box_1), np.array(box_2)

    x_start = np.maximum(box_1[0], box_2[0])
    y_start = np.maximum(box_1[1], box_2[1])
    x_end = np.minimum(box_1[2], box_2[2])
    y_end = np.minimum(box_1[3], box_2[3])

    area1 = (box_1[2] - box_1[0] + addone) * (box_1[3] - box_1[1] + addone)
    area2 = (box_2[2] - box_2[0] + addone) * (box_2[3] - box_2[1] + addone)
    overlap = np.maximum(x_end - x_start + addone, 0) * np.maximum(y_end - y_start + addone, 0)

    return overlap / min(area1, area2)


def nms_inter_classes(bboxes, iou_thres=0.3):
    """NMS between all classes

    Args:
        bboxes(list): [bboxes in cls1(np.array), bboxes in cls2(np.array), ...]. bboxes of each classes
        iou_thres(float): nsm threshold

    Returns:
        np.array: (n x 4).bboxes of targets after NMS between all classes
        list(list): (n x 1).labels of targets after NMS between all classes
    """

    lable_id = 0
    merge_bboxes, merge_labels = [], []
    for bboxes_cls in bboxes:
        if lable_id:
            merge_bboxes = np.concatenate((merge_bboxes, bboxes_cls), axis=0)
        else:
            merge_bboxes = bboxes_cls
        merge_labels += [[lable_id]] * len(bboxes_cls)
        lable_id += 1

    mark = np.ones(len(merge_bboxes), dtype=int)
    score_index = merge_bboxes[:, -1].argsort()[::-1]
    for i, cur in enumerate(score_index):
        if mark[cur] == 0:
            continue
        for ind in score_index[i + 1:]:
            if mark[ind] == 1 and rect_max_iou(merge_bboxes[cur], merge_bboxes[ind]) >= iou_thres:
                mark[ind] = 0
    new_bboxes = merge_bboxes[mark == 1, :4]
    new_labels = np.array(merge_labels)[mark == 1]
    new_labels = [list(map(int, lab)) for lab in new_labels]

    return new_bboxes, new_labels


def bbox2adj(bboxes_non):
    """Calculating row and column adjacent relationships according to bboxes of non-empty aligned cells

    Args:
        bboxes_non(np.array): (n x 4).bboxes of non-empty aligned cells

    Returns:
        np.array: (n x n).row adjacent relationships of non-empty aligned cells
        np.array: (n x n).column adjacent relationships of non-empty aligned cells
    """

    adjr = np.zeros([bboxes_non.shape[0], bboxes_non.shape[0]], dtype='int')
    adjc = np.zeros([bboxes_non.shape[0], bboxes_non.shape[0]], dtype='int')
    x_middle = bboxes_non[:, ::2].mean(axis=1)
    y_middle = bboxes_non[:, 1::2].mean(axis=1)
    for i, box in enumerate(bboxes_non):
        indexr = np.where((bboxes_non[:, 1] < y_middle[i]) & (bboxes_non[:, 3] > y_middle[i]))[0]
        indexc = np.where((bboxes_non[:, 0] < x_middle[i]) & (bboxes_non[:, 2] > x_middle[i]))[0]
        adjr[indexr, i], adjr[i, indexr] = 1, 1
        adjc[indexc, i], adjc[i, indexc] = 1, 1

        # Determine if there are special row relationship
        for j, box2 in enumerate(bboxes_non):
            if not (box2[1] + 4 >= box[3] or box[1] + 4 >= box2[3]):
                indexr2 = np.where((max(box[1], box2[1]) < y_middle[:]) & (y_middle[:] < min(box[3], box2[3])))[0]
                if len(indexr2):
                    adjr[j, i], adjr[i, j] = 1, 1

        # Determine if there are special column relationship
        for j, box2 in enumerate(bboxes_non):
            if not (box2[0] + 0 >= box[2] or box[0] + 0 >= box2[2]):
                indexc2 = np.where((max(box[0], box2[0]) < x_middle[:]) & (x_middle[:] < min(box[2], box2[2])))[0]
                if len(indexc2):
                    adjc[j, i], adjc[i, j] = 1, 1

    return adjr, adjc
