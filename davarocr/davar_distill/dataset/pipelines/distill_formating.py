"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    distill_formating.py
# Abstract       :    Definition of data formating process for knowledge distillation

# Current Version:    1.0.0
# Date           :    2022-07-07
##################################################################################################
"""
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formating import to_tensor, DefaultFormatBundle


@PIPELINES.register_module()
class DistillFormatBundle(DefaultFormatBundle):
    """ The common data format pipeline used by DavarCustom dataset. including,
        (1) transferred into Tensor (2) contained by DataContainer (3) put on device (GPU|CPU)

        - keys in ['img', 'gt_semantic_seg'] will be transferred into Tensor and put on GPU
        - keys in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore','gt_labels', 'stn_params']
          will be transferred into Tensor
        - keys in ['gt_masks', 'gt_poly_bboxes', 'gt_poly_bboxes_ignore', 'gt_cbboxes',
                   'gt_cbboxes_ignore', 'gt_texts', 'gt_text'] will be put on CPU
    """

    def __call__(self, results):
        for key in ['img', 'hr_img']:
            if key in results:
                img = results[key]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                results[key] = DC(to_tensor(img), stack=True)

        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'stn_params']:
            if key in results:
                results[key] = DC(to_tensor(results[key]))
                ori_key = 'hr_' + key
                if ori_key in results:
                    results[ori_key] = DC(to_tensor(results[ori_key]))

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        # Updated keys by DavarCustom dataset
        for key in ['gt_masks', 'gt_poly_bboxes', 'gt_poly_bboxes_ignore', 'gt_cbboxes',
                    'gt_cbboxes_ignore', 'gt_texts', 'gt_text', 'array_gt_texts', 'gt_bieo_labels']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)
            ori_key = 'hr_' + key
            if ori_key in results:
                results[ori_key] = DC(results[ori_key], cpu_only=True)

        return results
