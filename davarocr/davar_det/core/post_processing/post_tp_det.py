"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    points_generation.py
# Abstract       :    Generating fiducial points from predicted segmentation masks and regression masks

# Current Version:    1.0.0
# Date           :    2020-05-31
######################################################################################################
"""
import os
from ctypes import c_int, c_float, POINTER

import numpy as np
import numpy.ctypeslib as ctl


from davarocr.davar_common.core import POSTPROCESS
from .post_detector_base import BasePostDetector


@POSTPROCESS.register_module()
class TPPointsGeneration(BasePostDetector):
    """Point generation process in Shape Transform Module [1]

    Ref: [1] Text Perceptron: Towards End-to-End Arbitrary Shaped Text Spotting. AAAI-20.
                <https://arxiv.org/abs/2002.06820>`_
    """

    def __init__(self,
                 filter_ratio=0.5,
                 thres_text=0.45,
                 thres_head=0.5,
                 thres_bond=0.5,
                 point_num=14,
                 lib_name=None,
                 lib_dir="./",
                 ):
        """Point generation process in Shape Transform Module

        Args:
           filter_ratio(float):  filter out instances of which the boundaries are not sufficiently proper
           thres_text(float)  :  threshold for detecting center area
           thres_head(float)  :  threshold for detecting head and tail area
           thres_bond(float) :  threshold for detecting top and bottom boundaries
           point_num(int)  :  the number of fiducial points in the boundaries
           libname(str)      :  name of calling .so file
           libdir(str)       :  paht to calling .so file
        """
        super().__init__()

        # If there is no identified lib path, use the default path
        if lib_name is None or not os.path.isfile(os.path.join(lib_dir, lib_name)):
            cur_path = os.path.realpath(__file__)
            lib_dir = cur_path.replace('\\', '/').split('/')[:-1]
            lib_dir = "/".join(lib_dir) + '/lib'
            lib_name = "tp_points_generate.so"

        self.filter_ratio = filter_ratio
        self.thres_text = thres_text
        self.thres_head = thres_head
        self.thres_bond = thres_bond
        self.point_num = point_num

        assert 0.0 <= self.filter_ratio <= 1.0
        assert 0.0 <= self.thres_text <= 1.0
        assert 0.0 <= self.thres_head <= 1.0
        assert 0.0 <= self.thres_bond <= 1.0

        lib = ctl.load_library(lib_name, lib_dir)
        self.generate_func = lib.generate_result
        self.generate_func.argtypes = [ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # score_pred_text
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # score_pred_head
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # score_pred_tail
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # score_pred_bond
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # reg_head_pred
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # reg_tail_pred
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # reg_bond_pred
                                       c_int,  # height
                                       c_int,  # width
                                       c_int,  # pool ratio
                                       c_float,  # scale_factor
                                       c_int,  # point_num
                                       c_float,  # filter_ratio
                                       c_float,  # thres_text
                                       c_float,  # thres_head
                                       c_float,  # thres_bond
                                       ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),  # result
                                       POINTER(c_int)]  # result num

    def post_processing(self, mask_pred, img_meta):
        """ Do post-process;
            Here we only implement the fiducial points generation process in form of
            post-processing (with out loss back-propagation)

        Args:
            mask_pred(dict): predicted results in a dict, including
                             mask_pred['score_text_pred'], text/non-text classification mask
                             mask_pred['score_head_pred'], head/non-head classification mask
                             mask_pred['score_tail_pred'], tail/non-tail classification mask
                             mask_pred['score_bond_pred'], boundary/non-boundary classification mask
                             mask_pred['reg_head_pred'], regression predictions in head regions.
                             mask_pred['reg_tail_pred'], regression predictions in tail regions.
                             mask_pred['reg_bond_pred'], regression predictions in center-text
                                                         (offset to top&bottom boundary) regions.
            img_meta(dict): image meta-info
        Returns:
            list(dict): fiducial points (dict), in form of [{"points":[x1,y1, x2, y2, ..., xn, yn]}, ...]
        """

        # Get predicted feature maps
        score_pred_text = mask_pred['score_text_pred']
        score_pred_head = mask_pred['score_head_pred']
        score_pred_tail = mask_pred['score_tail_pred']
        score_pred_bond = mask_pred['score_bond_pred']
        reg_head_pred = mask_pred['reg_head_pred']
        reg_tail_pred = mask_pred['reg_tail_pred']
        reg_bond_pred = mask_pred['reg_bond_pred']

        # Used to store returned results
        results = []

        for i in range(score_pred_text.shape[0]):

            result = np.zeros((256, self.point_num * 2), dtype=np.int32)
            result_num = c_int()
            height = score_pred_text[i:i+1, :].shape[2]
            width = score_pred_text[i:i+1, :].shape[3]
            scale_factor = img_meta[i]["scale_factor"]
            if isinstance(scale_factor, (list, np.ndarray)):
                scale_factor = scale_factor[0]

            # Generate fiducial points by calling C++ lib
            self.generate_func(np.array(score_pred_text[i:i+1, :].cpu().detach()).reshape(-1),
                               np.array(score_pred_head[i:i+1, :].cpu().detach()).reshape(-1),
                               np.array(score_pred_tail[i:i+1, :].cpu().detach()).reshape(-1),
                               np.array(score_pred_bond[i:i+1, :].cpu().detach()).reshape(-1),
                               np.array(reg_head_pred[i:i+1, :].cpu().detach()).reshape(-1),
                               np.array(reg_tail_pred[i:i+1, :].cpu().detach()).reshape(-1),
                               np.array(reg_bond_pred[i:i+1, :].cpu().detach()).reshape(-1), height, width, 4,
                               scale_factor, self.point_num, self.filter_ratio, self.thres_text,
                               self.thres_head, self.thres_bond, result, result_num)
            result_tmp = dict()
            result_tmp["points"] = []
            for t in range(result_num.value):
                points = result[t]
                points2 = []

                for j in range(0, len(points), 2):
                    # Filter out points where corresponding element less than 0
                    if points[j] <= 0 and points[j] == points[j + 1]:
                        continue
                    points2.append(points[j])
                    points2.append(points[j + 1])
                result_tmp["points"].append(points2)
            results.append(result_tmp)
        return results
