"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    rcg_extra_aug.py
# Abstract       :    Implementations of Text Recognition Augmentation operations

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import numpy as np

from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class RcgExtraAugmentation:
    """ text recognition augmentation setting """

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_rotate=None,
                 complex_aug=None):
        """
        Args:
            photo_metric_distortion (dict): photo metric distortion setting
            expand (dict): image expand setting
            random_rotate (dict): random rotate setting
            complex_aug (dict): extra augmentation setting
        """

        # TODO: All of the Augmentation operations will be in the coming DavarOCR version

        self.transforms = []
        # photo metric distortion
        if photo_metric_distortion is not None:
            pass
            # self.transforms.append(PhotoMetricDistortion(**photo_metric_distortion))

        # expand
        if expand is not None:
            pass
            # self.transforms.append(Expand(**expand))

        # random rotate
        if random_rotate is not None:
            pass
            # self.transforms.append(RandomRotate(**random_rotate))

        # complex augmentation
        if complex_aug is not None:
            pass
            # self.transforms.append(ComplexAug(**complex_aug))

    def __call__(self, results):
        """
        Args:
            results (dict): training data dict

        Returns:
           dict: training data dict after augmentation
        """
        assert isinstance(results, dict) and 'img' in results

        img = results['img'].astype(np.float32)
        for transform in self.transforms:
            # augmentation transformation on image
            img = transform(img)
        results['img'] = img.astype(np.uint8)

        return results
