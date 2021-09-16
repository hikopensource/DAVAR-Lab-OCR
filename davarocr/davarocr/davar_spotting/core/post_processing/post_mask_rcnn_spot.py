"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    post_mask_rcnn_spot.py
# Abstract       :    Post processing of mask-rcnn based text spotter.
                      Get the contour of the mask area and format output.

# Current Version:    1.0.0
# Date           :    2021-09-01
####################################################################################################
"""
import cv2
import numpy as np

from davarocr.davar_common.core import POSTPROCESS

from .post_spotter_base import BasePostSpotter


@POSTPROCESS.register_module()
class PostMaskRCNNSpot(BasePostSpotter):
    """ Get the contour of the mask area and format output. """

    def __init__(self,
                 max_area_only=True,
                 use_rotated_box=False):
        """
        Args:
            max_area_only (boolean): whether to consider only one (maximum) region in each proposal regions.
            use_rotated_box (boolean): whether to use minAreaRect to represent text regions (or use contour polygon)
        """

        super().__init__()
        self.max_area_only = max_area_only
        self.use_rotated_box = use_rotated_box

    def approx_poly(self, mask):
        """ Get contour of mask regions.
        Args:
            mask (list(list(boolean)): Bitmap mask

        Returns:
            list(list(int)): polygon contours of mask, e.g. [[x1_1,y1_1,x1_2,y1_2, ... ],
                             [x2_1, y2_1, x2_2, y2_2, ...],...]
        """

        mask_expand = mask.copy()
        contours, _ = cv2.findContours(mask_expand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        approx_curve = []
        if self.max_area_only:
            contour_areas = [cv2.contourArea(contour) for contour in contours]
            if len(contour_areas) == 0:
                return []
            max_index = np.argmax(np.array(contour_areas))
            max_contour = contours[max_index]
            if self.use_rotated_box:
                # In minimum rotated rectangle
                min_rect = cv2.minAreaRect(max_contour)
                poly = cv2.boxPoints(min_rect)
                poly = np.int0(poly)
            else:
                # In polygon contours
                perimeter = cv2.arcLength(max_contour, True) * 0.01
                poly = cv2.approxPolyDP(max_contour, perimeter, True)
            approx_curve.append(poly)
        else:
            for contour in contours:
                perimeter = cv2.arcLength(contour, True) * 0.01
                poly = cv2.approxPolyDP(contour, perimeter, True)
                approx_curve.append(poly)
        return approx_curve

    def post_processing(self, batch_result, img_metas, **kwargs):
        """
        Args:
            batch_result (dict):forward output needs to be formatted, including keys:
                               'seg_preds',  predict masks, list(np.array(N, H, W)), len = B
                               'bboxes_preds', predict bboxes, list(np.array(N, 8)), len = B
                               'text_preds', predict transcriptions, list(["text1", ...],...[]), len = B
            img_metas (dict): image meta info
            **kwargs: other parameters

        Returns:
            list(dict): Format results, like [{'points':[], 'texts':[]}]
        """
        seg_preds = batch_result['seg_preds']
        text_preds = batch_result['text_preds']
        det_results = []

        for batch_id, batch_seg in enumerate(seg_preds):
            det_result = dict()
            batch_text = text_preds[batch_id]
            batch_img_meta = img_metas[batch_id]
            w_scale, h_scale = batch_img_meta['scale_factor'][:2]

            det_result['points'] = []
            det_result['texts'] = []

            for box_id, seg in enumerate(batch_seg):
                text = batch_text[box_id]
                seg = np.array(seg[:, :, np.newaxis], dtype='uint8')
                curve_poly = self.approx_poly(seg)
                if len(curve_poly) == 0:
                    continue

                curve_poly = curve_poly[0].squeeze()

                # Filter out curve poly with less than 2 points.
                curve_poly = curve_poly.astype(np.int)
                if len(curve_poly.shape) < 2:
                    continue

                curve_poly = curve_poly.reshape(-1)
                curve_poly[0::2] = curve_poly[0::2] / w_scale
                curve_poly[1::2] = curve_poly[1::2] / h_scale

                det_result['points'].append(curve_poly.tolist())
                det_result['texts'].append(text)

            det_results.append(det_result)

        return det_results
