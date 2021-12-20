"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    seg_det_formatting.py
# Abstract       :    Definition of seg_based_method data formationg

# Current Version:    1.0.0
# Date           :    2020-05-31
###################################################################################################
"""

import numpy as np

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class SegFormatBundle():
    """Format returned results, and convert gt_masks to tensor and store in Data Container
        img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
        gt_masks: (1)to tensor, (2)to DataContainer (stack=True)

    Args:
        results(dict): Data flow

    Returns:
         dict: Formated data flow
    """

    def __call__(self, results):

        if 'img' in results:
            img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img).float(), stack=True)
        if 'gt_masks' in results:
            results['gt_masks'] = DC(to_tensor(results['gt_masks']), stack=True)
        if 'gt_texts' in results:
            results['gt_texts'] = DC(results['gt_texts'], cpu_only=True)
        if 'gt_poly_bboxes' in results:
            results['gt_poly_bboxes'] = DC(results['gt_poly_bboxes'], cpu_only=True)

        return results

    def __repr__(self):
        return self.__class__.__name__
