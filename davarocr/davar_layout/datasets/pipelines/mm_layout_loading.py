"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mm_layout_loading.py
# Abstract       :    MMLA_LoadAnnotations definition for mm_layout_analysis.

# Current Version:    1.0.0
# Date           :    2020-12-06
##################################################################################################
"""
import numpy as np

from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from davarocr.davar_common.datasets import DavarLoadAnnotations


@PIPELINES.register_module()
class MMLALoadAnnotations(DavarLoadAnnotations):
    def __init__(self,
                 with_bbox=False,
                 with_poly_bbox=False,
                 with_poly_mask=False,
                 with_care=False,
                 with_label=False,
                 with_multi_label=False,
                 with_text=False,
                 with_cbbox=False,
                 text_profile=None,
                 label_start_index=0,
                 poly2mask=True,
                 bieo_labels=None,

                 #mmla_loading
                 with_cattribute=False,
                 with_ctexts=False,
                 with_bbox_2=False,
                 with_poly_mask_2=False,
                 with_label_2=False,
                 custom_classes=None,
                 custom_classes_2=None,
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
            with_cattribute (boolean): Whether to load attribute in character granularity.
            with_ctexts (boolean):     Whether to load character sequence annotations.
            with_bbox_2 (boolean):     Whether to load bbox annotation in larger scope (e.g. layout compared to text).
            with_poly_mask_2 (boolean):Whether to load poly mask annotation in larger scope (e.g. layout compared to
                                      text).
            with_label_2 (boolean):    Whether to load bbox labels in larger scope (e.g. layout compared to text).
            custom_classes (list[int]):  custom specified classes in finest scope (e.g. text lines).
            custom_classes_2 (list[int]):custom specified classes in larger scope (e.g. layout components).
        """
        self.with_cattribute = with_cattribute
        self.with_ctexts = with_ctexts
        self.with_bbox_2 = with_bbox_2
        self.with_poly_mask_2 = with_poly_mask_2
        self.with_label_2 = with_label_2
        self.custom_classes = custom_classes
        self.custom_classes_2 = custom_classes_2

        super().__init__(
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

    def _load_cares(self, results):
        """ Load and parse results['cares'], process ann_info and ann_info_2 separately.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['cares'] will be updated.
        """
        for per_ann in ['ann_info', 'ann_info_2']:
            if per_ann in results.keys():
                ann = results[per_ann]
                bboxes_length = len(ann['bboxes']) if 'bboxes' in ann else 1

                if self.with_care:
                    cares = np.array(ann.get('cares', np.ones(bboxes_length)))
                else:
                    cares = np.ones(bboxes_length)

                ann["cares"] = cares

        return results

    def _load_bboxes(self, results, ann_idx=1):
        """ Load and parse results['bboxes'], bboxes are labeled in form of axis-aligned bounding boxes, e.g.,
                    [[x_min, y_min, x_max, y_max],...].
            Process ann_info and ann_info_2 separately.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['bboxes'] and results['bboxes_ignore'] will be updated.
        """
        if ann_idx == 1:
            ann = results['ann_info']
        else:
            ann = results['ann_info_2']
        cares = ann['cares']

        tmp_gt_bboxes = ann['bboxes']

        gt_bboxes = []
        gt_bboxes_ignore = []
        for i, box in enumerate(tmp_gt_bboxes):
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

        if len(gt_bboxes) == 0:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)

        if len(gt_bboxes_ignore) == 0:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)

        if ann_idx == 1:
            results['gt_bboxes'] = gt_bboxes
            results['gt_bboxes_ignore'] = gt_bboxes_ignore

            results['bbox_fields'].append('gt_bboxes')
            results['bbox_fields'].append('gt_bboxes_ignore')
        else:
            results['gt_bboxes_2'] = gt_bboxes
            results['gt_bboxes_ignore_2'] = gt_bboxes_ignore

            results['bbox_fields'].append('gt_bboxes_2')
            results['bbox_fields'].append('gt_bboxes_ignore_2')

        return results

    def _load_polymasks(self, results, ann_idx=1):
        """Private function to load mask annotations. Process ann_info and ann_info_2 separately.

        Args:
            results (dict): Result dict from :obj:`DavarCustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations. If ``self.poly2mask`` is set ``True``,
                  `gt_mask` will contain:obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """
        height, width = results['img_info']['height'], results['img_info']['width']
        if ann_idx == 1:
            gt_masks = results['ann_info']['segboxes']
        else:
            cares = results['ann_info_2']['cares']
            ori_masks = results['ann_info_2']['segboxes']
            gt_masks = []
            for idx in range(len(ori_masks)):
                if cares[idx] == 1:
                    gt_masks.append(ori_masks[idx])

        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, height, width) for mask in gt_masks], height, width)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], height,
                width)

        if ann_idx == 1:
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')
        else:
            results['gt_masks_2'] = gt_masks
            results['mask_fields'].append('gt_masks_2')

        return results

    def _load_labels(self, results, ann_idx=1):
        """Load and parse results['labels']. Process ann_info and ann_info_2 separately.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['gt_labels'] will be updated, in shape of [1, 2, 3, 2, 1, ...].
        """
        if ann_idx == 1:
            ann = results['ann_info']
        else:
            ann = results['ann_info_2']

        tmp_labels = ann.get("labels", None)
        if tmp_labels is None:
            if "bboxes" in ann:
                tmp_labels = [[1]] *  len(ann["bboxes"])
            else:
                tmp_labels = [[1]]

        gt_labels=[]
        gt_labels_ignore = []
        cares = ann['cares']

        # only collect care==1 labels.
        for i, label in enumerate(tmp_labels):
            if cares[i] == 1:
                gt_labels.append(label[0])
            else:
                gt_labels_ignore.append(label[0])

        # If `labels` are in string type, we transfer them into integers, and then add it with the start index.
        if len(gt_labels) > 0 and isinstance(gt_labels[0], str):
            assert results['classes_config'] is not None
            assert 'classes' in results['classes_config'] or 'classes_0' in results['classes_config']
            for i, _ in enumerate(gt_labels):
                if 'classes_0' in results['classes_config']:
                    classes_config = results['classes_config']["classes_0"]
                else:
                    classes_config = results['classes_config']['classes']

                if self.label_start_index == -1:
                    # When label_start_index is -1, means `notLabel` or `NotLabeled` is the
                    # first class in classes_config.
                    assert ('NotLabeled' in classes_config) or ('NoLabel' in classes_config)
                    if 'NotLabeled' in classes_config:
                        notlabeled_index = classes_config.index('NotLabeled')
                    else:
                        notlabeled_index = classes_config.index('NoLabel')

                    # Swap the position between NotLabel and -1
                    if notlabeled_index > 0:
                        classes_config[notlabeled_index], classes_config[0] = \
                            classes_config[0], classes_config[notlabeled_index]

                # Add labels with start_index
                gt_labels[i] = classes_config.index(gt_labels[i]) + self.label_start_index

        # If `labels_ignore` are in string type, we transfer them into integers, and then add it with the start index.
        if len(gt_labels_ignore) > 0 and isinstance(gt_labels_ignore[0], str):
            assert results['classes_config'] is not None
            assert 'classes' in results['classes_config'] or 'classes_0' in results['classes_config']
            for i, _ in enumerate(gt_labels_ignore):
                if 'classes_0' in results['classes_config']:
                    classes_config = results['classes_config']["classes_0"]
                else:
                    classes_config = results['classes_config']['classes']

                if self.label_start_index == -1:
                    # When label_start_index is -1, means `notLabel` or `NotLabeled` is the
                    # first class in classes_config.
                    assert ('NotLabeled' in classes_config) or ('NoLabel' in classes_config)
                    if 'NotLabeled' in classes_config:
                        notlabeled_index = classes_config.index('NotLabeled')
                    else:
                        notlabeled_index = classes_config.index('NoLabel')

                    # Swap the position between NotLabel and -1
                    if notlabeled_index > 0:
                        classes_config[notlabeled_index], classes_config[0] = \
                            classes_config[0], classes_config[notlabeled_index]

                # Add labels_ignore with start_index
                gt_labels_ignore[i] = classes_config.index(gt_labels_ignore[i]) + self.label_start_index

        if ann_idx == 1:
            results['gt_labels'] = gt_labels
            results['gt_labels_ignore'] = gt_labels_ignore
        else:
            results['gt_labels_2'] = gt_labels
            results['gt_labels_ignore_2'] = gt_labels_ignore

        return results

    def _load_texts(self, results):
        """ Load and parse character level bounding boxes from results['texts'].

        Args:
           results(dict): Data flow used in DavarCustomDataset.

        Returns:
           dict: updated data flow. results['gt_texts'] (list[str]) and results['gt_text'] (str) will be updated,

        """
        ann = results['ann_info']
        tmp_gt_texts = []
        tmp_texts = ann.get('texts', [])
        cares = ann['cares']

        for i, text in enumerate(tmp_texts):
            if self.filtered:
                text = [c for c in text if c in self.character]
                text = ''.join(text)

            # Transfer text according to the sensitive tag.
            if self.sensitive == 'upper':
                text = text.upper()
            elif self.sensitive == 'lower':
                text = text.lower()

            if len(text) > self.text_max_length:
                # cares[i] = 0
                text = text[:self.text_max_length]
            if cares[i] == 1:
                tmp_gt_texts.append(text)

        ann['cares'] = cares
        results['gt_texts'] = tmp_gt_texts

        if len(results['gt_texts']) == 1:
            results['gt_text'] = tmp_gt_texts[0]

        return results

    def _filter_custom_classes_data(self, results, ann_idx=1):
        """ Filter annotations of custom specified classes.

        Args:
           results(dict): Data flow used in DavarCustomDataset.

        Returns:
           dict: updated data flow.

        """
        if ann_idx == 1:
            ann = results['ann_info']
            custom_classes_list = self.custom_classes
        else:
            ann = results['ann_info_2']
            custom_classes_list = self.custom_classes_2

        cares = ann['cares']

        tmp_labels = ann.get("labels", None)

        for idx, per_label in enumerate(tmp_labels):
            if per_label[0] in custom_classes_list:
                continue
            else:
                cares[idx] = 0

        if ann_idx == 1:
            results['ann_info']['cares'] = cares
        else:
            results['ann_info_2']['cares'] = cares
        return results

    def _load_cbboxes(self, results):
        """ Load and parse character level bounding boxes from results['cbboxes'].

        Args:
           results(dict): Data flow used in DavarCustomDataset.

        Returns:
           dict: updated data flow. results['gt_cbboxes'] and results['gt_cbboxes_ignore'] will be updated,
                 in shape of 3-dimensional vectors [[[x1, y1, ..., x4, y4],[],[]], [[],[]] ...].
        """
        ann = results['ann_info']
        cares = ann['cares']
        assert "cbboxes" in ann

        tmp_cbboxes = ann.get('cbboxes', [])
        tmp_gt_cbboxes = []
        tmp_gt_cbboxes_ignore = []
        for i, cbboxes in enumerate(tmp_cbboxes):
            if cares[i] == 1:
                tmp_gt_cbboxes.append(cbboxes)
            else:
                tmp_gt_cbboxes_ignore.append(cbboxes)

        results['gt_cbboxes'] = tmp_gt_cbboxes
        results['gt_cbboxes_ignore'] = tmp_gt_cbboxes_ignore
        results['cbbox_fields'].append('gt_cbboxes')
        results['cbbox_fields'].append('gt_cbboxes_ignore')

        return results

    def _load_cattribute(self, results):
        """Load attribute of each characters.

        Args:
           results(dict): Data flow used in DavarCustomDataset.

        Returns:
           dict: updated data flow.

        """
        ann = results['ann_info']
        cattributes = ann.get('cattributes', [])
        results['gt_cattributes'] = cattributes
        return results

    def _load_ctexts(self, results):
        """Load character sequences.

        Args:
           results(dict): Data flow used in DavarCustomDataset.

        Returns:
           dict: updated data flow.

        """
        ann = results['ann_info']
        tmp_texts = ann.get('ctexts', [])
        results['gt_ctexts'] = tmp_texts
        return results

    def __call__(self, results):
        """ Main process.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: output data flow.
        """
        results = self._load_cares(results)

        if self.custom_classes is not None:
            results = self._filter_custom_classes_data(results)

        # Loading text might affect the value of `gt_cares`
        if self.with_text:
            results = self._load_texts(results)
        if self.with_bbox:
            results = self._load_bboxes(results)
        if self.with_label:
            results = self._load_labels(results)
        if self.with_cbbox:
            results = self._load_cbboxes(results)
        if self.with_cattribute:
            results = self._load_cattribute(results)
        if self.with_ctexts:
            results = self._load_ctexts(results)

        # ann2
        if self.custom_classes_2 is not None:
            results = self._filter_custom_classes_data(results, ann_idx=2)
        if self.with_bbox_2:
            results = self._load_bboxes(results, ann_idx=2)
        if self.with_poly_mask_2:
            results = self._load_polymasks(results, ann_idx=2)
        if self.with_label_2:
            results = self._load_labels(results, ann_idx=2)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_poly_bbox={}, with_poly_mask={},with_care={}, with_label={}, '
                     'with_multi_lalbel={}, with_text={}, with_cbbox={}, with_cbbox_labels={}').format(
            self.with_bbox, self.with_poly_bbox, self.with_poly_mask, self.with_care, self.with_label,
            self.with_multi_label, self.with_text, self.with_cbbox, self.with_cbbox_labels)
        return repr_str
