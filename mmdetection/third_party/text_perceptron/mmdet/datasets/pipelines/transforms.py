"""
###################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    transforms.py
# Abstract       :    Customize the resize transformation and define the random rotate transformation

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31

# Modified Date  :    2020-11-25
# Modified by    :    inusheng
# Comments       :    Code and comment standardized
#####################################################################################################
"""
from math import fabs, sin, cos, radians
import numpy as np

import cv2
from mmdet.datasets.registry import PIPELINES
from mmdet.datasets.pipelines import Resize

@PIPELINES.register_module
class DavarResize(Resize):
    """
    Description:
        randomly scaling images and corresponding annotations;
        extended implementation to support resizing polygon boxes (polygon boxes may cause arrays of different lengths.)
    
    Property:
        see in mmdet/datasets/pipelines/transforms.py
    """
    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            if isinstance(results[key], np.ndarray):
                bboxes = results[key] * results['scale_factor']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
                results[key] = bboxes
            else:
                # if annotations are stored as np.array, do resize operation seperately on every element. 
                bboxes = []
                for box in results[key]:
                    tmp_box = np.array(box) * results['scale_factor']
                    tmp_box[0::2] = np.clip(tmp_box[0::2], 0, img_shape[1] - 1)
                    tmp_box[1::2] = np.clip(tmp_box[1::2], 0, img_shape[0] - 1)
                    bboxes.append(tmp_box)
                results[key] = bboxes


@PIPELINES.register_module
class RandomRotate(object):
    """
    Description:
        randomly rotate images and corresponding annotations
        
        angles: contains single value or multiple values
                if angles contains single value, this value represents `rotating fixed angle`;
                if angles contains two values, angles represents `rotating random angle in the interval ranged by these two values`
                if angles contains more than two values, angles represents `randomly choosing a value in this list as rotation angle`

    Property:
        angles(int/tuple/list(tuple)): rotation angles
    """

    def __init__(self,
                 angles=None):
        self.angles = angles

    def _rotate_img(self, results):
        """
        Description
            randomly rotating images and corresponding annotations
        
        Args:
            results: data stream in pipeline (dict; img and bboxes_fields are frequently used keys)
        
        Returns:
            results: data stream in pipeline (dict)
        """
        angle = self.angle
        height, width = results['img_shape'][:2]

        # compute height and width of new image after rotation
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        # compute rotation matrix
        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2

        # rotating image
        imgRotation = cv2.warpAffine(results['img'], matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        results['img'] = imgRotation

        # rotating corresponding annotations
        for key in results.get('bbox_fields', []):
            gt_boxes_ret = []
            for poly in results[key]:
                rot_array = []
                # convert to np.array of shape (:, 2)
                for i in range(0, len(poly), 2):
                    rot_array.append(np.array([int(poly[i]), int(poly[i + 1])]))
                rot_array = np.array([rot_array])

                # rotating corresponding annotations
                rot_array = cv2.transform(rot_array, matRotation).squeeze().reshape(len(poly))
                gt_boxes_ret.append(rot_array)
            results[key] = gt_boxes_ret

    def __call__(self, results):
        # angles must be a list or tuple
        assert isinstance(self.angles, (list, tuple))
        if len(self.angles) == 1:
            angle = self.angles[0]
        elif len(self.angles) == 2:
            angle_max = max(self.angles)
            angle_min = min(self.angles)
            angle = np.random.randint(angle_min, angle_max)
        else:
            angle = np.random.choice(self.angles)
        self.angle = angle
        self._rotate_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(angles={})'.format(self.angles)
        return repr_str