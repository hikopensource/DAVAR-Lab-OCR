"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_loading.py
# Abstract       :    Definition of tabular data loading, corresponding to TabularCustomDataset

# Current Version:    1.0.1
# Date           :    2021-09-18
##################################################################################################
"""

import numpy as np
from mmdet.datasets.builder import PIPELINES
from davarocr.davar_common.datasets.pipelines import DavarLoadAnnotations
from davarocr.davar_table.core import recon_noncell, recon_largecell


@PIPELINES.register_module()
class DavarLoadTableAnnotations(DavarLoadAnnotations):
    """The common loading function, used by DavarTableDataset. An example is as follows:

        test_datalist.json:
        {
            "Images/test/1110.0169v1.15.png": {
                "height": 90,
                "width": 395,
                "content_ann": {
                    "bboxes": [ [],
                                [78, 10, 116, 22],
                                [],
                                [252, 10, 338, 22],
                                [426, 10, 512, 22],
                                [10, 26, 44, 39],
                                [82, 26, 116, 39], ...],
                    "texts": [
                        "", "Out-", "", "Dataset1", "Dataset2", "Dim", "liers", ...
                    ],
                    "cells": [
                        [0, 0, 0, 0],[0, 1, 0, 1],[0, 2, 0, 2],[0, 3, 0, 4],[0, 5, 0, 6],[1, 0, 1, 0],[1, 1, 1, 1], ...
                    ],
                    "labels": [
                        ["t-head"],["t-head"],["t-head"],["t-head"],["t-head"],["t-body"],["t-body"], ...
                    ]
                }
            },
            ....
        },
    """

    def __init__(self,
                 with_bbox=False,
                 with_poly_bbox=False,
                 with_poly_mask=False,
                 with_care=False,
                 with_label=False,
                 with_multi_label=False,
                 with_text=False,
                 with_cbbox=False,
                 bieo_labels=None,
                 text_profile=None,
                 label_start_index=0,
                 poly2mask=True,
                 with_enlarge_bbox=False,
                 with_empty_bbox=False,
                 ):
        """ Parameter initialization

        Args:
            with_bbox(boolean):       Whether to parse and load the bbox annotation. Used in situations that boxes are
                                      labeled as 2-point format, e.g., [[x_min, y_min, x_max, y_max],...],
                                      `results['gt_bboxes']` and `results['gt_bboxes_ignore']` will be updated.
            with_poly_bbox(boolean):  Whether to parse and load the bbox annotation. Used in situations that boxes are
                                      labeled as 2N-point format, e.g., [[x_1, y_1, x_2, y_2,..., x_n, y_n],...],
                                      `results['gt_poly_bboxes']` & `results['gt_poly_bboxes_ignore']` will be updated.
            with_poly_mask(boolean):  Whether to parse and load the mask annotation according to 'bboxes'.
                                      `results['gt_masks']` will be updated.
            with_care(boolean):       Whether to parse and load NOT_CARE label. The 'care' label decide whether a bbox
                                      is participant in training. e.g. [1, 0, 1, 1]. `results['cares']` will be updated.
            with_label(boolean):      Whether to parse and load labels (labels can be int or strings). Used in
                                      situations that labels are in 1-dimensional vectors as [[1], [3], [4], [2], ...].
                                      `results['gt_labels']"`and `results['gt_labels_ignore']` will be updated.
            with_multi_label(boolean):Whether to parse and load multiple labels (labels can be int or strings). Used in
                                      situations that labels are labeled in N-dimensional vectors like
                                      [[1,2], [3,5], [4,6], [2,2], ...]. `results['gt_labels']` and
                                      `results['gt_labels_ignore']` will be updated.
            with_text(boolean):       Whether to parse and load text transcriptions. e.g., ["apple", "banana", ...]
                                      `results['gt_texts']` and `results['gt_text']` will be updated.
            with_cbbox(boolean):      Whether to parse and load the character-wised bbox annotation.
                                      e.g., [[[x1, y1, x2, y2, x3, y3, x4, y4], [] ,[]], [[],[],[]]...],
                                      `results['gt_cbboxes']` and `results['gt_cbboxes_ignore']` will be updated.
            text_profile(dict):       Configuration of text loading, including:.
                                      text_profile=dict(
                                         text_max_length=25,   # maximum suppored text length
                                         sensitive='upper',    # whether to transfer character into "upper" or "lower" ,
                                                                 default "same"
                                         filtered=True,        # whether to filter out unsupported characters
                                         character="abcdefg"   # support character list
                                      )
            label_start_index(list[int]):When gt_labels are labeled in `str` type, we will transfer them into `int` type
                                     according to `classes_config`. The start label will be added. e.g., for mmdet 1.x,
                                     this value is set to [1];  for mmdet 2.x, this will be set to [0].
            poly2mask (boolean):      Whether to convert the instance masks from polygons to bitmaps. Default: True.
            with_enlarge_bbox (boolean):    Whether to parse and load the enlarge bbox annotation. Default: Fasle.
            with_empty_bbox (boolean):  Whether to parse and load the empty bbox annotation. Default: Fasle.
        """

        super(DavarLoadTableAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_poly_bbox=with_poly_bbox,
            with_poly_mask=with_poly_mask,
            with_care=with_care,
            with_label=with_label,
            with_multi_label=with_multi_label,
            with_text=with_text,
            with_cbbox=with_cbbox,
            bieo_labels=bieo_labels,
            text_profile=text_profile,
            label_start_index=label_start_index,
            poly2mask=poly2mask
        )

        self.with_enlarge_bbox = with_enlarge_bbox
        self.with_empty_bbox = with_empty_bbox

    def _load_enlarge_bboxes(self, results):
        """ Load and parse results['bboxes'], bboxes are labeled in form of axis-aligned bounding boxes, e.g.,
            [[x_min, y_min, x_max, y_max],...].

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['gt_bboxes'] and results['gt_bboxes_ignore'] will be updated.
        """

        ann = results['ann_info']
        cares = results['cares']

        tmp_gt_bboxes = ann.get('bboxes', [])
        tmp_gt_cells = ann.get('cells', [])

        # Generating pseudo labels for bboxes of cells using bboxes of contents and corresponding row/col span
        tmp_gt_enlarge_bboxes = recon_largecell(tmp_gt_bboxes, tmp_gt_cells)

        gt_bboxes = []
        gt_bboxes_ignore = []
        for i, box in enumerate(tmp_gt_enlarge_bboxes):
            box_i = np.array(box, dtype=np.double)
            x_coords = box_i[0::2]
            y_coords = box_i[1::2]
            aligned_box = [
                np.min(x_coords),
                np.min(y_coords),
                np.max(x_coords),
                np.max(y_coords)
            ]
            if cares[i] == 1:
                gt_bboxes.append(aligned_box)
            else:
                gt_bboxes_ignore.append(aligned_box)

        # If there is no bboxes in an image, we fill the results with a np array in shape of (0, 4)
        if len(gt_bboxes) == 0:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)

        if len(gt_bboxes_ignore) == 0:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)

        if self.with_bbox:
            results['gt_content_bboxes'] = results['gt_bboxes']
            results['gt_content_bboxes_ignore'] = results['gt_bboxes_ignore']
            results['bbox_fields'].append('gt_content_bboxes')
            results['bbox_fields'].append('gt_content_bboxes_ignore')
        else:
            results['bbox_fields'].append('gt_bboxes')
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_ignore'] = gt_bboxes_ignore

        return results

    def _load_empty_bboxes(self, results):
        """ Generate pseudo labels for empty cells

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['gt_empty_bboxes'] and results['gt_empty_bboxes_ignore'] will be updated.
        """

        ann = results['ann_info']
        empty_cares = [care for box, care in zip(ann['bboxes'], results['cares']) if not box]

        # Generate pseudo labels for bboxes of empty cells
        high, width = results['img_info']['height'], results['img_info']['width']
        tmp_gt_enlarge_bboxes = recon_noncell(ann['bboxes'], ann['cells'], (high, width))
        tmp_gt_empty_bboxes = [empty for box, empty in zip(ann['bboxes'], tmp_gt_enlarge_bboxes) if not box]

        gt_empty_bboxes = []
        gt_empty_bboxes_ignore = []
        for i, box in enumerate(tmp_gt_empty_bboxes):
            if empty_cares[i] == 1:
                gt_empty_bboxes.append(box)
            else:
                gt_empty_bboxes_ignore.append(box)

        # If there is no bboxes in an image, we fill the results with a np array in shape of (0, 4)
        if len(gt_empty_bboxes) == 0:
            gt_empty_bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_empty_bboxes = np.array(gt_empty_bboxes, dtype=np.float32)
        if len(gt_empty_bboxes_ignore) == 0:
            gt_empty_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_empty_bboxes_ignore = np.array(gt_empty_bboxes_ignore, dtype=np.float32)

        results['gt_empty_bboxes'] = gt_empty_bboxes
        results['gt_empty_bboxes_ignore'] = gt_empty_bboxes_ignore
        results['bbox_fields'].append('gt_empty_bboxes')
        results['bbox_fields'].append('gt_empty_bboxes_ignore')

        return results

    def _filter_empty(self, results):
        """ Filter empty cells without bbox annotation

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['cares'] and results['ann_info'] will be updated.
        """

        ann = results['ann_info']
        assert len(ann.get('bboxes', [])) and len(ann.get('cells', []))

        # Convert cells with illegal bboxes to empty cells
        for i, box in enumerate(ann['bboxes']):
            if not box:
                continue
            box_i = np.array(box, dtype=np.double)
            x_coords = box_i[0::2]
            y_coords = box_i[1::2]
            if np.max(x_coords) <= np.min(x_coords) or np.max(y_coords) <= np.min(y_coords):
                results['ann_info']['bboxes'][i] = []

        if self.with_empty_bbox:
            results = self._load_empty_bboxes(results)

        # filter targets without bbox annotation
        valid_cells = [cell for box, cell in zip(ann['bboxes'], ann['cells']) if box]
        results['ann_info']['cells'] = valid_cells
        valid_cares = [care for box, care in zip(ann['bboxes'], results['cares']) if box]
        results['cares'] = valid_cares
        if "labels" in ann:
            valid_labels = [lab for box, lab in zip(ann['bboxes'], ann['labels']) if box]
            results['ann_info']['labels'] = valid_labels
        valid_bboxes = [box for box in ann['bboxes'] if box]
        results['ann_info']['bboxes'] = valid_bboxes

        return results

    def __call__(self, results):
        """ Main process

        Args:
            results(dict): Data flow used in DavarCustomDataset

        Returns:
            results(dict): Data flow used in DavarCustomDataset
        """

        assert 'ann_info' in results
        # Cares will be used for all tasks.
        results = self._load_cares(results)

        # Filtering empty cells. If with_empty_bbox is true, produce pseudo-labels for empty cells.
        results = self._filter_empty(results)

        # Loading text might affect the value of `gt_cares`
        if self.with_text:
            results = self._load_texts(results)
        if self.with_poly_bbox:
            results = self._load_poly_bboxes(results)
        if self.with_bbox:
            results = self._load_bboxes(results)

        # Loading enlarge_bbox
        if self.with_enlarge_bbox:
            results = self._load_enlarge_bboxes(results)

        if self.with_poly_mask:
            results = self._load_polymasks(results)
        if self.with_label:
            results = self._load_labels(results)
        if self.with_multi_label:
            results = self._load_multi_labels(results)
        if self.with_cbbox:
            results = self._load_cbboxes(results)
        if self.bieo_labels is not None and self.with_box_bieo_labels:
            results = self._load_box_bieo_labels(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            '(with_bbox={}, with_poly_bbox={}, with_poly_mask={}, '
            'with_care={}, with_label={}, with_multi_lalbel={}, '
            'with_text={}, with_cbbox={}, with_enlarge_bbox={}, with_empty_bbox={}').format(
            self.with_bbox, self.with_poly_bbox, self.with_poly_mask,
            self.with_care, self.with_label, self.with_multi_label,
            self.with_text, self.with_cbbox, self.with_enlarge_bbox, self.with_empty_bbox)
        return repr_str
