"""
###################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    transforms.py
# Abstract       :    Implementations of some transformations

# Current Version:    1.0.1
# Date           :    2020-11-25
#####################################################################################################
"""
from math import fabs, sin, cos, radians
import numpy as np
import cv2
import mmcv

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Resize
import torchvision.transforms as transforms
from PIL import Image


@PIPELINES.register_module()
class DavarResize(Resize):
    """ Resize images & bbox & mask. Add new specialities of
        - support poly boxes resize
        - support cbboxes resize
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
                                          keep_ratio=keep_ratio, bbox_clip_border=bbox_clip_border, backend=backend,
                                          override=override)

    def _resize_bboxes(self, results):
        """ Resize bboxes (support 'gt_bboxes', 'gt_poly_bboxes').
            Refactor this function to support multiple points

        Args:
            results(dict): input data flow

        Returns:
            dict: updated data flow. All keys in `bbox_fields` will be updated according to `scale_factor`.
        """
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            if isinstance(results[key], np.ndarray) and (len(results[key] > 0) and len(results[key][0]) == 4):
                # For 2-point rect bounding boxes, np.ndarray(K, 4)
                bboxes = results[key] * results['scale_factor']
                if self.bbox_clip_border:
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                results[key] = bboxes
            else:
                # For any n-point poly bounding boxes.
                bboxes = []
                for box in results[key]:
                    tmp_box = np.array(box, dtype=np.float32)
                    tmp_box[0::2] *= results['scale_factor'][0]
                    tmp_box[1::2] *= results['scale_factor'][1]
                    if self.bbox_clip_border:
                        tmp_box[0::2] = np.clip(tmp_box[0::2], 0, img_shape[1])
                        tmp_box[1::2] = np.clip(tmp_box[1::2], 0, img_shape[0])
                    bboxes.append(tmp_box)
                results[key] = bboxes

    def _resize_cbboxes(self, results):
        """ Resize cbboxes (support 'gt_cbboxes').

        Args:
            results(dict): input data flow

        Returns:
            dict: updated data flow. All keys in `cbbox_fields` will be updated according to `scale_factor`.
        """
        img_shape = results['img_shape']
        for key in results.get('cbbox_fields', []):
            if isinstance(results[key], np.ndarray):
                # For 2-point rect bounding boxes, np.ndarray(K, L, 4)
                cbboxes = results[key] * results['scale_factor']
                if self.bbox_clip_border:
                    cbboxes[:, :, 0::2] = np.clip(cbboxes[:, :, 0::2], 0, img_shape[1])
                    cbboxes[:, :, 1::2] = np.clip(cbboxes[:, :, 1::2], 0, img_shape[0])
                results[key] = cbboxes
            else:
                # For n-point poly bounding boxes, LIST[ [ [[2*N]], [2*N]], [...] ..., [2*KN]]
                cbboxes = []
                for cbox in results[key]:
                    tmp_cbox = np.array(cbox, dtype=np.float32)
                    if tmp_cbox.shape[1] == 4:
                        tmp_cbox = tmp_cbox * results['scale_factor']
                    else:
                        new_tmp_cbox = []
                        for ccbox in tmp_cbox:
                            ccbox = np.array(ccbox, dtype=np.float32)
                            ccbox[0::2] *= results['scale_factor'][0]
                            ccbox[1::2] *= results['scale_factor'][1]
                            new_tmp_cbox.append(ccbox)
                        tmp_cbox = np.array(new_tmp_cbox, dtype=np.float32)
                    if self.bbox_clip_border:
                        tmp_cbox[:, 0::2] = np.clip(tmp_cbox[:, 0::2], 0, img_shape[1])
                        tmp_cbox[:, 1::2] = np.clip(tmp_cbox[:, 1::2], 0, img_shape[0])
                    cbboxes.append(tmp_cbox)
                results[key] = cbboxes

    def __call__(self, results):
        """ Main process of davar_resize

        Args:
            results(dict): input data flow.

        Returns:
            dict: updated data flow.
        """
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple([int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, 'scale and scale_factor cannot be both set.'
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_cbboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)

        return results


@PIPELINES.register_module()
class RandomRotate():
    """Randomly rotate images and corresponding annotations"""

    def __init__(self,
                 angles=None,
                 borderValue=(255, 255, 255)
                 ):
        """
        Args:
            angles(list | tuple):   the rotated degree, contains a single value or multiple values
                                    - If `angles` contrains single value, this value represents `rotating fixed angle`;
                                    - If `angles` contains two values(tuple or lis), angles represents `rotating
                                        random angle in the interval ranged by these two values`
                                    -  If angles contains more than two values, angles represents `randomly choosing
                                        a value in this list as rotation angle`
            borderValue(tuple): filled pixels value when rotate.
        """
        self.angles = angles
        self.angle = 0
        self.borderValue = borderValue

    def _rotate_img(self, results):
        """Randomly rotating images and corresponding annotations

        Args:
            results(dict): data flow in pipeline (contains `img` and `bboxes_fields`)

        Returns:
            dict: updated data flow in pipeline
        """
        angle = self.angle
        height, width = results['img_shape'][:2]

        # Compute height and width of new image after rotation
        height_new = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        width_new = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        # Compute rotation matrix
        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        mat_rotation[0, 2] += (width_new - width) / 2
        mat_rotation[1, 2] += (height_new - height) / 2

        # Rotate image
        img_rotation = cv2.warpAffine(results['img'], mat_rotation, (width_new, height_new),
                                      borderValue=self.borderValue)
        results['img'] = img_rotation

        # Rotate corresponding annotations: all boxes in bbox_fields
        for key in results.get('bbox_fields', []):
            gt_boxes_ret = []
            for poly in results[key]:
                rot_array = []
                # Convert to np.array of shape (:, 2)
                for i in range(0, len(poly), 2):
                    rot_array.append(np.array([int(poly[i]), int(poly[i + 1])]))
                rot_array = np.array([rot_array])

                # Rotate corresponding annotations
                rot_array = cv2.transform(rot_array, mat_rotation).squeeze().reshape(len(poly))
                gt_boxes_ret.append(rot_array)
            results[key] = gt_boxes_ret

        # Edit gt_bboxes according to gt_poly_bboxes
        if "gt_bboxes" in results["bbox_fields"] and "gt_poly_bboxes" in results["bbox_fields"]:
            gt_poly_bboxes = results["gt_poly_bboxes"]
            gt_poly_bboxes_ignore = results["gt_poly_bboxes_ignore"]
            gt_bboxes = []
            gt_bboxes_ignore = []

            for poly_box in gt_poly_bboxes:
                x_coords = poly_box[0::2]
                y_coords = poly_box[1::2]
                aligned_box = [
                    np.min(x_coords),
                    np.min(y_coords),
                    np.max(x_coords),
                    np.max(y_coords)
                ]
                gt_bboxes.append(aligned_box)

            for poly_box_ignore in gt_poly_bboxes_ignore:
                x_coords = poly_box_ignore[0::2]
                y_coords = poly_box_ignore[1::2]
                aligned_box = [
                    np.min(x_coords),
                    np.min(y_coords),
                    np.max(x_coords),
                    np.max(y_coords)
                ]
                gt_bboxes_ignore.append(aligned_box)

            results["gt_bboxes"] = gt_bboxes
            results["gt_bboxes_ignore"] = gt_bboxes_ignore

        # Rotate corresponding annotations: all boxes in cbbox_fields
        for key in results.get('cbbox_fields', []):
            gt_cboxes_ret = []
            for instance in results[key]:
                tmp_cboxes = []
                for poly in instance:
                    rot_array = []
                    # Convert to np.array of shape (:, 2)
                    for i in range(0, len(poly), 2):
                        rot_array.append(np.array([int(poly[i]), int(poly[i + 1])]))
                    rot_array = np.array([rot_array])

                    # Rotate corresponding annotations
                    rot_array = cv2.transform(rot_array, mat_rotation).squeeze().reshape(len(poly))
                    tmp_cboxes.append(rot_array)
                    gt_cboxes_ret.append(tmp_cboxes)
            results[key] = gt_cboxes_ret

        # Rotate corresponding annotations: all masks in mask_fields
        for key in results.get('mask_fields', []):
            mask = results[key].masks.transpose((1, 2, 0))
            # Rotate mask
            mask_rotation = cv2.warpAffine(mask, mat_rotation, (width_new, height_new),
                                          borderValue=self.borderValue)
            if mask_rotation.ndim == 2:
                # case when only one mask, (h, w)
                mask_rotation = mask_rotation[:, :, None]  # (h, w, 1)
            mask_rotation = mask_rotation.transpose((2, 0, 1))
            results[key].masks = mask_rotation
            results[key].height = height_new
            results[key].width = width_new

    def __call__(self, results):
        """ Main process of davar_resize

        Args:
            results(dict): input data flow

        Returns:
            dict: updated data flow, `img` and  boxes in `bbox_fields` and `cbbox_fields' will be transformed.
        """
        # `angles` must be a list or tuple
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


@PIPELINES.register_module()
class ColorJitter:
    """An interface for torch color jitter so that it can be invoked in mmdetection pipeline."""

    def __init__(self, **kwargs):
        self.transform = transforms.ColorJitter(**kwargs)

    def __call__(self, results):
        """Main process of color jitter, refer to mmocr

        Args:
            results(dict): input data flow

        Returns:
            dict: updated data flow, key of `img` will be update

        """
        # Image is bgr
        img = results['img'][..., ::-1]
        img = Image.fromarray(img)
        img = self.transform(img)
        img = np.asarray(img)
        img = img[..., ::-1]
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ResizeNormalize:
    """ Package resize normalize in a pipeline, Support for different interpolate modes"""
    def __init__(self, size,
                 interpolation=2,
                 mean=(127.5, 127.5, 127.5),
                 std=(127.5, 127.5, 127.5),):
        """
        Args:
            size (tuple): image resize size
            interpolation (int): interpolation type, including [0, 1, 2, 3]
            mean (tuple): image normalization mean
            std (tuple): image normalization std
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, results):
        """
        Args:
            results (dict): dict with training image

        Returns:
            dict: dict with training image after resize and normalization
        """
        img = results['img']

        # different interpolation type corresponding the OpenCV
        if self.interpolation == 0:
            self.interpolation = cv2.INTER_NEAREST
        elif self.interpolation == 1:
            self.interpolation = cv2.INTER_LINEAR
        elif self.interpolation == 2:
            self.interpolation = cv2.INTER_CUBIC
        elif self.interpolation == 3:
            self.interpolation = cv2.INTER_AREA
        else:
            raise Exception("Unsupported interpolation type !!!")

        # Deal with the image error during image loading
        if img is None:
            return None
        try:
            img = cv2.resize(img, self.size, self.interpolation)
        except cv2.error:
            return None
        img = np.array(img, np.float32)

        # normalize the image
        img = mmcv.imnormalize(img, self.mean, self.std, to_rgb=False)
        results['img'] = img
        return results
