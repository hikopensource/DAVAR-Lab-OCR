"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    points_generation.py
# Abstract       :    Generating fiducial points from predicted segmentation masks and regression masks

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31

# Modified Date  :    2020-11-26
# Modified by    :    inusheng
# Comments       :    Code and comment standardized
######################################################################################################
"""
from ctypes import *

import numpy as np
import numpy.ctypeslib as ctl
import os
import torch
import torch.nn as nn

from mmdet.models.registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class PointsGeneration(nn.Module):
    """
    Description:
        point generation process in Shape Transform Module
    
    Arguments:
        filter_ratio:  filter out instances of which the boundaries are not sufficiently proper 
        thres_text  :  threshold for detecting center area
        thres_head  :  threshold for detecting head and tail area 
        thres_bond  :  threshold for detecting top and bottom boundaries 
        point_num   :  the number of fiducial points in the boundaries 
        libname     :  name of calling .so file 
        libdir      :  paht to calling .so file 

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
        # If there is no identified lib path, use the default path
        if lib_name is None or not os.path.isfile(os.path.join(lib_dir, lib_name)):
            cur_path = os.path.realpath(__file__)
            lib_dir = cur_path.replace('\\', '/').split('/')[:-1]
            lib_dir = "/".join(lib_dir) + '/lib'
            lib_name = "tp_data.so"

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
        self.generate_func.argtypes = [ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),    # score_pred_text
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),    # score_pred_head
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),    # score_pred_tail
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),    # score_pred_bond
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),    # reg_head_pred
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),    # reg_tail_pred
                                       ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),    # reg_bond_pred
                                       c_int,                                              # height
                                       c_int,                                              # width
                                       c_int,                                              # pool ratio
                                       c_float,                                            # scale_factor
                                       c_int,                                              # point_num
                                       c_float,                                            # filter_ratio
                                       c_float,                                            # thres_text
                                       c_float,                                            # thres_head
                                       c_float,                                            # thres_bond
                                       ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),      # result
                                       POINTER(c_int)]                                     # result num

    def forward(self, mask_pred, img_meta):
        """
        Description:
            network forward pass;
            NOTE: here we only implemented to compute the fiducial points, not end-to-end forward functionality

        Arguments:
            mask_pred：predicted results (tuple) 
            img_meta：image meta-info
        Returns:
            results：fiducial points (dict), {"points":[x1,y1, x2, y2, ..., xn, yn]}
        """

        # get predicted feature maps 
        score_pred_text_4, score_pred_head_4, score_pred_tail_4, score_pred_bond_4, reg_head_pred,  reg_tail_pred, reg_bond_pred = mask_pred

        # feature maps normalization 
        score_pred_text = torch.sigmoid(score_pred_text_4)
        score_pred_head = torch.sigmoid(score_pred_head_4)
        score_pred_tail = torch.sigmoid(score_pred_tail_4)
        score_pred_bond = torch.sigmoid(score_pred_bond_4)

        # stor returned results 
        result = np.zeros((256, self.point_num * 2), dtype=np.int32)
        result_num = c_int()

        height = score_pred_text.shape[2]
        width = score_pred_text.shape[3]

        # generate fiducial points by calling C++ lib 
        self.generate_func(np.array(score_pred_text.cpu().detach()).reshape(-1),
                           np.array(score_pred_head.cpu().detach()).reshape(-1),
                           np.array(score_pred_tail.cpu().detach()).reshape(-1),
                           np.array(score_pred_bond.cpu().detach()).reshape(-1),
                           np.array(reg_head_pred.cpu().detach()).reshape(-1),
                           np.array(reg_tail_pred.cpu().detach()).reshape(-1),
                           np.array(reg_bond_pred.cpu().detach()).reshape(-1), height, width, 4,
                           img_meta[0]["scale_factor"], self.point_num, self.filter_ratio, self.thres_text,
                           self.thres_head, self.thres_bond, result, result_num)

        results = dict()
        results["points"] = []
        for i in range(result_num.value):
            points = result[i]
            points2 = []

            for j in range(0, len(points), 2):
                # filter out points where corresponding element less than 0 
                if points[j] <= 0 and points[j] == points[j + 1]:
                    continue
                points2.append(points[j])
                points2.append(points[j + 1])
            results["points"].append(points2)
        return results

