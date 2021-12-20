"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    loading.py
# Abstract       :    Definition of video text detection data formating process

# Current Version:    1.0.0
# Date           :    2021-06-02
##################################################################################################
"""

from math import fabs, sin, cos, radians

import cv2
import numpy as np

from mmdet.datasets.pipelines import Pad, Normalize
from mmdet.datasets.builder import PIPELINES
from davarocr.davar_common.datasets.pipelines import DavarResize, RandomRotate, ResizeNormalize, ColorJitter


@PIPELINES.register_module()
class ConsistResize(DavarResize):
    """ Same with DavarResize, the only difference is ConsistResize support results(list) contain multiple
    instance and  resize to same size for all the instances in results(list)

    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        """
        Args:
            img_scale (tuple or list[tuple]): Images scales for resizing.
            multiscale_mode (str): Either "range" or "value".
            ratio_range (tuple[float]): (min_ratio, max_ratio)
            keep_ratio (bool): Whether to keep the aspect ratio when resizing the
                image.
            bbox_clip_border (bool, optional): Whether clip the objects outside
                the border of the image. Defaults to True.
            backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
                These two backends generates slightly different results. Defaults
                to 'cv2'.
            override (bool, optional): Whether to override `scale` and
                `scale_factor` so as to call resize twice. Default False. If True,
                after the first resizing, the existed `scale` and `scale_factor`
                will be ignored so the second resizing can be allowed.
                This option is a work-around for multiple times of resize in DETR.
                Defaults to False.
        """

        super().__init__(img_scale=img_scale, multiscale_mode=multiscale_mode, ratio_range=ratio_range,
                         keep_ratio=keep_ratio, bbox_clip_border=bbox_clip_border, backend=backend, override=override)

    def __call__(self, results):
        """ Main process of ConsistResize

        Args:
            results(dict | list(dict)): input data flow.

        Returns:
            dict | list(dict): updated data flow.
        """

        # Deal with results(dict) contains single instance
        if isinstance(results, dict):
            results = super().__call__(results)
            return results

        # Deal with results(list(dict)) contains multiple instances
        scale = None
        scale_idx = None
        results_ = []
        for instance in results:

            # Use same scale to resize all instances in results(list(dict))
            if scale:
                instance['scale'] = scale
                instance['scale_idx'] = scale_idx
            instance = super().__call__(instance)
            scale = instance['scale']
            scale_idx = instance['scale_idx']
            results_.append(instance)

        return results_


@PIPELINES.register_module()
class ConsistRandomRotate(RandomRotate):
    """same with RandomRotate, the only difference is ConsistRandomRotate support results(list) contain multiple
    instance and  resize to same size for all the instances in results(list)
    """

    def __init__(self, angles=None):
        """
        Args:
            angles(list | tuple):   the rotated degree, contains a single value or multiple values
                                    - If `angles` contrains single value, this value represents `rotating fixed angle`;
                                    - If `angles` contains two values(tuple or lis), angles represents `rotating
                                        random angle in the interval ranged by these two values`
                                    -  If angles contains more than two values, angles represents `randomly choosing
                                        a value in this list as rotation angle`
        """
        super().__init__(angles)

    def _rotate_flow(self, results):
        """Randomly rotating optical flow

        Args:
            results(dict): data flow in pipeline

        Returns:
            dict: updated data flow in pipeline
        """

        angle = self.angle
        result_flows = []
        if 'flow' in results['img_info']:
            flows = results['img_info']['flow']
            for k in range(flows.shape[0]):
                h_flow, w_flow, _ = flows[k].shape

                # Compute height and width of new image after rotation
                height_new = int(w_flow * fabs(sin(radians(angle))) + h_flow * fabs(cos(radians(angle))))
                width_new = int(h_flow * fabs(sin(radians(angle))) + w_flow * fabs(cos(radians(angle))))

                # Compute rotation matrix
                mat_rotation = cv2.getRotationMatrix2D((w_flow / 2, h_flow / 2), angle, 1)
                mat_rotation[0, 2] += (width_new - w_flow) / 2
                mat_rotation[1, 2] += (height_new - h_flow) / 2

                # Rotating flow
                flow_rotation = cv2.warpAffine(flows[k], mat_rotation, (width_new, height_new),
                                               borderValue=(0, 0, 0))
                result_flows.append(flow_rotation)

            results['img_info']['flow'] = np.stack(result_flows)

    def __call__(self, results):
        """ Main process of ConsistRandomRotate

        Args:
            results(dict): input data flow

        Returns:
            dict or list(dict): updated data flow, `img` , boxes, flow will be transformed.
        """
        # Angles must be a list or tuple
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

        # Deal with results(dict) contains single instance
        if isinstance(results, dict):
            results = super().__call__(results)
            return results

        # Deal with results(list(dict)) contains multiple instances
        results_ = []
        for instance in results:

            # Rotate img and gt_bboxes
            self._rotate_img(instance)

            # Rotate the optical flow
            self._rotate_flow(instance)
            results_.append(instance)
        return results_


@PIPELINES.register_module()
class ConsistColorJitter(ColorJitter):
    """Same with ColorJitter, the only difference is ConsistColorJitter support results(list) contain multiple
    instance and  do same operation for all the instances in results(list)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, results):
        """Main process of color jitter, refer to mmocr

        Args:
            results(dict or list): input data flow

        Returns:
            results(dict or list): updated data flow, key of `img` will be update

        """
        # Deal with results(dict) contains single instance
        if isinstance(results, dict):
            results = super().__call__(results)
            return results

        # Deal with results(list(dict)) contains multiple instances
        results_ = []
        for instance in results:
            instance = super().__call__(instance)
            results_.append(instance)
        return results_


@PIPELINES.register_module()
class ConsistResizeNormalize(ResizeNormalize):
    """ Same with ResizeNormalize, the only difference is ConsistResizeNormalize support results(list) contain multiple
    instance and  do same operation for all the instances in results(list) """

    def __init__(self, size,
                 interpolation=0,
                 mean=(127.5, 127.5, 127.5),
                 std=(127.5, 127.5, 127.5),):
        """
        Args:
            size (tuple): image resize size
            interpolation (int): interpolation type, including [0, 1, 2, 3]
            mean (tuple): image normalization mean
            std (tuple): image normalization std
        """
        super().__init__(size=size, interpolation=interpolation, mean=mean, std=std)

    def __call__(self, results):
        """
        Args:
            results (dict or list(dict)): images

        Returns:
            dict or list(dict): dict with images after resize and normalization
        """
        # Deal with results(dict) contains single instance
        if isinstance(results, dict):
            results = super().__call__(results)
            return results

        # Deal with results(list(dict)) contains multiple instances
        results_ = []
        for instance in results:
            instance = super().__call__(instance)
            results_.append(instance)
        return results


@PIPELINES.register_module()
class ConsistPad(Pad):
    """ Same with Pad, the only difference is ConsistPad support results(list) contain multiple instance and do same
    operation for all the instances in results(list)
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        super().__init__(size=size, size_divisor=size_divisor, pad_val=pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict | list(dict)): Results from loading pipeline.

        Returns:
            dict | list(dict): Updated results.

        """
        # Deal with results(dict) contains single instance
        if isinstance(results, dict):
            results = super().__call__(results)
            return results

        # Deal with results(list(dict)) contains multiple instances
        results_ = []
        for instance in results:
            instance = super().__call__(instance)
            results_.append(instance)

        return results_


@PIPELINES.register_module()
class ConsistNormalize(Normalize):
    """ Same with Normalize. the only difference is ConsistNormalize support results(list) contain multiple instance
    and do same operation for all the instances in results(list)
    """

    def __init__(self, mean, std, to_rgb=True):
        super().__init__(mean=mean, std=std, to_rgb=to_rgb)

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict | list(dict)): Results from loading pipeline.

        Returns:
            dict | list(dict): Normalized results, 'img_norm_cfg' key is added into result dict.

        """
        # Deal with results(dict) contains single instance
        if isinstance(results, dict):
            results = super().__call__(results)
            return results

        # Deal with results(list(dict)) contains multiple instances
        results_ = []
        for instance in results:
            instance = super().__call__(instance)
            results_.append(instance)
        return results_
