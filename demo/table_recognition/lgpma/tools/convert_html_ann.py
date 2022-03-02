"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    convert_html_to_ann.py
# Abstract       :    Converting html in original PubTabNet to training format(davar format)

# Current Version:    1.0.1
# Date           :    2022-01-26
##################################################################################################
"""

import numpy as np


def html_to_davar(html):
    """ Converting html in original PubTabNet to content_ann in training format(davar format)

    Args:
        html (dict): "html" in original PubTabNet, which is used to describe table structure and content. For example:
                    "html": {
                            "cells": [
                                {"tokens": ["<b>", "T", "r", "a", "i", "t", "</b>"],
                                 "bbox": [11, 5, 33, 14]},
                                {"tokens": ["<b>", "M", "e", "a", "n", "</b>"],
                                 "bbox": [202, 5, 225, 14]},
                                {"tokens": ["S", "C", "S"],
                                 "bbox": [14, 27, 30, 35]},
                                {"tokens": ["-", " ", "0", ".", "1", "0", "2", "4"],
                                 "bbox": [199, 27, 229, 35]}
                            ],
                            "structure": {"tokens":
                                ["<thead>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</thead>",
                                 "<tbody>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</tbody>"]
                            }
                    }
    Returns:
        dict:   "content_ann" in ours training dataset (davar format), like:
                "content_ann": {
                                "bboxes": [[11, 5, 33, 14],
                                           [202, 5, 225, 14],
                                           [14, 27, 30, 35],
                                           [199, 27, 229, 35]],

                                "texts": ["Trait", "Mean", "SCS", "- 0.1024"],

                                "texts_tokens": [["<b>", "T", "r", "a", "i", "t", "</b>"],
                                                 ["<b>", "M", "e", "a", "n", "</b>"],
                                                 ["S", "C", "S"],
                                                 ["-", " ", "0", ".", "1", "0", "2", "4"]],

                                "cells": [[0,0,0,0],
                                          [0,1,0,1],
                                          [1,0,1,0],
                                          [1,1,1,1]],

                                "labels": [[0],    # [0] for thead and [1] for tbody
                                           [0],
                                           [1],
                                           [1]]
                 }
    """
    assert len(html['cells']) == html['structure']['tokens'].count('</td>')
    bboxes, texts, texts_tokens, cells, labels = [], [], [], [], []

    # get cells and labels using span_matrix and number of head
    span_matrix = get_matrix(html['structure']['tokens'])  # get span_matrix represent table structure
    num_h, num_b = get_headbody(html['structure']['tokens'])  # get number of t-head and t-body
    for i in range(1, 1 + span_matrix.max()):
        where = np.where(span_matrix == i)
        s_r, e_r = int(where[0].min()), int(where[0].max())
        s_c, e_c = int(where[1].min()), int(where[1].max())
        cells.append([s_r, s_c, e_r, e_c])
        #labels.append(["t-head"]) if i <= num_h else labels.append(["t-body"])
        labels.append([0]) if i <= num_h else labels.append([1])

    # get bboxes, texts and texts_tokens
    charsign = ['<b>', '<i>', '<sup>', '<sub>', '<underline>', '</b>', '</i>', '</sup>', '</sub>', '</underline>']
    for cell in html['cells']:
        bbox = cell.get('bbox', [])
        # filter out the bbox with area 0
        if bbox and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0:
            bboxes.append(bbox)
            text = ''.join([t for t in cell["tokens"] if t not in charsign])
            texts.append(text)
            texts_tokens.append(cell["tokens"])
        else:
            bboxes.append([])
            texts.append("")
            texts_tokens.append([])

    content_ann = {"bboxes": bboxes, "texts": texts, "texts_tokens": texts_tokens, "cells": cells, "labels": labels}

    return content_ann


def get_matrix(html_str):
    """Convert html to span matrix, a two-dimensional matrix representing table structure

    Args:
        html_str(str): html representing table structure.

    Returns:
        np.array(num_row x num_col): span matrix
    """

    # get number of rows and colunms from html
    row_index = [i for i, tag in enumerate(html_str) if tag == "<tr>"]
    num_row = len(row_index)

    num_col = 0
    html_row0 = html_str[row_index[0]:row_index[1]]
    for ind, tag in enumerate(html_row0):
        if "rowspan" in tag:
            # if "rowspan" and "colspan" appears togehetr, skip curent tag("rowspan")
            if ind != len(html_row0) - 1 and "colspan" in html_row0[ind + 1]:
                continue
            else:
                num_col += 1
        elif "colspan" in tag:
            num_col += int(tag[-3:-1]) if tag[-3:-1].isdigit() else int(tag[-2])
        elif tag == "<td>":
            num_col += 1

    # convert html to span matrix, a two-dimensional matrix representing table structure
    span_matrix = np.zeros([num_row, num_col], dtype=int) - 1
    staus = html_to_area(html_str, row_index, span_matrix)
    if staus:  # if html is illegal, return zeros
        span_matrix = np.zeros([num_row, num_col], dtype=int)

    return span_matrix


def get_headbody(html_str):
    """Calculating number of bboxes belonging to "t-head" and "t-body"  respectively

    Args:
        html_str(str): html representing table structure

    Returns:
        int: number of bboxes belonging to "t-head"
        int: number of bboxes belonging to "t-body"
    """
    s_h, e_h = html_str.index('<thead>'), html_str.index('</thead>')
    s_b, e_b = html_str.index('<tbody>'), html_str.index('</tbody>')
    num_h = html_str[s_h + 1:e_h].count('</td>')
    num_b = html_str[s_b + 1:e_b].count('</td>')
    return num_h, num_b


def html_to_area(html_str, row_index, span_matrix):
    """Convert html to span matrix, a two-dimensional matrix representing table structure

    Args:
        html_str(str): html representing table structure.
        row_index(list): index of each row in html.
        span_matrix(np.array): a two-dimensional matrix representing table structure.

    Returns:
        np.array(num_row x num_col): span matrix
    """

    num_row, num_col = span_matrix.shape[0], span_matrix.shape[1]

    staus = 0  # whether the given html is illegal
    area_index = 1
    row_index.append(len(html_str))
    for i in range(num_row):
        col_index = 0  # record column number of the current row
        spantogether = 0
        html_cur_row = html_str[row_index[i]:row_index[i + 1]]

        for ind, tag in enumerate(html_cur_row):
            if spantogether:
                spantogether = 0
                continue
            # if cur tag is not key information,continue
            if tag != "<td>" and "span" not in tag:
                continue

            if col_index > num_col - 1:
                return 1  # The column of current row exceeds the column of the first row
            # current cell is a part of row span cell
            while span_matrix[i, col_index] != -1:
                if col_index == num_col - 1:
                    return 1
                else:
                    col_index += 1

            # basic cell
            if tag == "<td>":
                span_matrix[i, col_index] = area_index
                col_index += 1
            # "rowspan" and "colspan" together
            elif "rowspan" in tag and (ind != len(html_cur_row) - 1 and "colspan" in html_cur_row[ind + 1]):
                row = int(tag[-3:-1]) if tag[-3:-1].isdigit() else int(tag[-2])
                col = int(html_cur_row[ind + 1][-3:-1]) \
                    if html_cur_row[ind + 1][-3:-1].isdigit() else int(html_cur_row[ind + 1][-2])
                spantogether = 1  # the next span will be skipped
                if (span_matrix[i:i + row, col_index:col_index + col] != -1).any():
                    return 3  # Overlay between cells
                span_matrix[i:i + row, col_index:col_index + col] = area_index
                if i + row > span_matrix.shape[0] or col_index + col > span_matrix.shape[1]:
                    return 2  # Spanning cell exceeds the table boundary
                col_index += col
            # only "colspan"
            elif "colspan" in tag:
                col = int(tag[-3:-1]) if tag[-3:-1].isdigit() else int(tag[-2])
                if col_index + col > num_col:
                    return 2
                if (span_matrix[i, col_index:col_index + col] != -1).any():
                    return 3
                span_matrix[i, col_index:col_index + col] = area_index
                col_index += col
            # only "rowspan"
            elif "rowspan" in tag:
                row = int(tag[-3:-1]) if tag[-3:-1].isdigit() else int(tag[-2])
                if i + row > num_row:
                    return 2
                if (span_matrix[i:i + row, col_index] != -1).any():
                    return 3
                span_matrix[i:i + row, col_index] = area_index
                col_index += 1
            area_index += 1
    if -1 in span_matrix:
        staus = 1  # The column number of some rows is smaller than the column number of the first row

    return staus
