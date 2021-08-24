"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    post_mask_rcnn.py
# Abstract       :    Post processing of Mask-RCNN text detector.
                      Get the contour of the mask area and format output.

# Current Version:    1.0.0
# Date           :    2020-05-31
######################################################################################################
"""
import numpy as np
import cv2
from davarocr.davar_common.core import POSTPROCESS
from .post_detector_base import BasePostDetector


@POSTPROCESS.register_module()
class PostMaskRCNN(BasePostDetector):
    """ Get the contour of the mask area and format output"""
    def __init__(self,
                 max_area_only=True,
                 use_rotated_box=False
                 ):
        """

        Args:
            max_area_only(boolean): whether to consider only one (maximum) region in each proposal regions.
            use_rotated_box(boolean): whether to use minAreaRect to represent text regions (or use contour polygon)
        """
        super().__init__()
        self.max_area_only = max_area_only
        self.use_rotated_box = use_rotated_box

    def approx_poly(self, mask):
        """ Get contour of mask regions

        Args:
            mask(list(list(boolean)): Bitmap mask

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
                poly = cv2.approxPolyDP(max_contour, 3, True)
            approx_curve.append(poly)
        else:
            for contour in contours:
                poly = cv2.approxPolyDP(contour, 3, True)
                approx_curve.append(poly)
        return approx_curve

    def post_processing(self, batch_result, **kwargs):
        """
        Args:
            batch_result(list(Tensor)): prediction results, [(box_result, seg_result), ...]
            **kwargs: other parameters

        Returns:
            list(dict): Format results, like [{'points':[[x1, y1, ..., xn, yn],[],...], 'confidence':[0.9,0.8,...]}]
        """

        det_results = []

        for result in batch_result:
            det_result = dict()
            box_result, seg_result = result
            det_result['points'] = []
            det_result['confidence'] = []
            det_result['labels'] = []
            for i in range(len(box_result)):
                boxes_pred = box_result[i]
                seg_pred = seg_result[i]
                assert boxes_pred.shape[0] == len(seg_pred)
                for box_id in range(boxes_pred.shape[0]):
                    prob = boxes_pred[box_id, 4]
                    seg = seg_pred[box_id]
                    seg = np.array(seg[:,:, np.newaxis], dtype='uint8')

                    curve_poly = self.approx_poly(seg)
                    if len(curve_poly) == 0:
                        continue
                    curve_poly = curve_poly[0].squeeze()
                    if len(curve_poly.shape) < 2:
                        continue

                    curve_poly = curve_poly.astype(np.int)
                    curve_poly = curve_poly.reshape(-1).tolist()
                    det_result['points'].append(curve_poly)
                    det_result['confidence'].append(prob)
                    det_result['labels'].append([i])
            det_results.append(det_result)

        return det_results
