"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_formating.py
# Abstract       :    Definition of common data formating process

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formating import to_tensor, DefaultFormatBundle


@PIPELINES.register_module()
class DavarCollect():
    """ Collect specific data from the data flow (results)"""
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                            'img_norm_cfg')):
        """

        Args:
            keys(list[str]): keys that need to be collected
            meta_keys(tuple): keys of img_meta that need to be collected. e.g.,
                            - "img_shape": image shape, (h, w, c).
                            - "scale_factor": the scale factor of the re-sized image to the original image
                            - "flip": whether the image is flipped
                            - "filename": path to the image
                            - "ori_shape": original image shape
                            - "pad_shape": image shape after padding
                            - "img_norm_cfg": configuration of normalizations
        """
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """ Main process of davar_collect

        Args:
            results(dict): input data flow

        Returns:
            dict: collected data informations from original data flow
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        # Add feature to support situation without img_metas.
        if len(img_metas) != 0:
            data['img_metas'] = DC(img_metas, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
            self.keys, self.meta_keys)


@PIPELINES.register_module()
class DavarDefaultFormatBundle(DefaultFormatBundle):
    """ The common data format pipeline used by DavarCustom dataset. including, (1) transferred into Tensor
        (2) contained by DataContainer (3) put on device (GPU|CPU)

        - keys in ['img', 'gt_semantic_seg'] will be transferred into Tensor and put on GPU
        - keys in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore','gt_labels', 'stn_params']
          will be transferred into Tensor
        - keys in ['gt_masks', 'gt_poly_bboxes', 'gt_poly_bboxes_ignore', 'gt_cbboxes', 'gt_cbboxes_ignore',
                  'gt_texts', 'gt_text'] will be put on CPU
    """

    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'stn_params']:
            if key not in results:
                continue

            results[key] = DC(to_tensor(results[key]))

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        # Updated keys by DavarCustom dataset
        for key in ['gt_masks', 'gt_poly_bboxes', 'gt_poly_bboxes_ignore', 'gt_cbboxes',
                    'gt_cbboxes_ignore', 'gt_texts', 'gt_text', 'array_gt_texts', 'gt_bieo_labels']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)

        return results
