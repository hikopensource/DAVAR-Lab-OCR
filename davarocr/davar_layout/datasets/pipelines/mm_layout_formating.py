"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mm_layout_formating.py
# Abstract       :    format bundle for mm_layout_analysis.

# Current Version:    1.0.0
# Date           :    2020-12-06
##################################################################################################
"""
import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class MMLAFormatBundle():
    """Add gt_bboxes_2 etc to support content_ann_2.
    """
    def __call__(self, results):
        """Data format bundle, including, (1) transferred into Tensor, (2) contained by DataContainer (3) put on device
         (GPU|CPU).

         Args:
         	results (dict): k-v pairs of data, e.g. 'gt_labels': [1,2,3]

        Returns:
        	dict: formatted data dict.
        """
        if 'img' in results:
            if isinstance(results['img'], DC):
                pass
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img).float(), stack=True)

        # support content_ann_2
        for key in ['gt_bboxes_2', 'gt_bboxes_ignore_2', 'gt_labels_2', 'proposals', 'gt_bboxes', 'gt_bboxes_ignore',
                    'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        for key in ['gt_masks', 'gt_masks_2']:
            if key not in results:
                continue
            results[key] = DC(results[key], cpu_only=True)

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        for key in ['input_ids', 'token_type_ids', 'attention_mask', 'gt_ctexts', 'gt_cattributes', 'in_bboxes_2',
                    'gt_cbboxes', 'gt_texts']:
            if key not in results:
                continue
            results[key] = DC(results[key], cpu_only=True)

        return results

    def __repr__(self):
        return self.__class__.__name__
