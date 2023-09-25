"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ctunet_loading.py
# Abstract       :    Definition of ctunet dataset loading process

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""

from mmdet.datasets.builder import PIPELINES
from davarocr.davar_common.datasets.pipelines import DavarLoadAnnotations


@PIPELINES.register_module()
class CTUNetLoadAnnotations(DavarLoadAnnotations):
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
                 with_relations=False,
                 with_rowcols=False
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
            with_relations (boolean): Whether to parse and load the relations of cells. Default: Fasle.
            with_rowcols (boolean): Whether to parse and load the row/column numbers of cells. Default: Fasle.
        """

        super(CTUNetLoadAnnotations, self).__init__(
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

        self.with_relations = with_relations
        self.with_rowcols = with_rowcols

    def _load_relations(self, results):
        ann = results['ann_info']
        if 'relations' in ann:
            results['relations'] = ann['relations']
        else:
            results['relations'] = []

        return results

    def _load_rowcols(self, results):
        ann = results['ann_info']
        results['gt_rowcols'] = ann['cells']

        return results

    def __call__(self, results):
        """ Main process.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: output data flow.
        """
        assert 'ann_info' in results
        # Cares will be used for all tasks.
        results = self._load_cares(results)

        # Loading text might affect the value of `gt_cares`
        if self.with_text:
            results = self._load_texts(results)
        if self.with_poly_bbox:
            results = self._load_poly_bboxes(results)
        if self.with_bbox:
            results = self._load_bboxes(results)
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

        if self.with_relations:
            results = self._load_relations(results)
        if self.with_rowcols:
            results = self._load_rowcols(results)

        return results
