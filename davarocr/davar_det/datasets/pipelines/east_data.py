"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    east_data.py
# Abstract       :    EAST ground truth generation

# Current Version:    1.0.0
# Date           :    2021-06-08
##################################################################################################
"""
import os
import math
from ctypes import c_int, c_float
import numpy as np
from numpy import random
import numpy.ctypeslib as ctl

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class EASTDataGeneration:
    """EAST training data generation [1]

    Ref: [1] An Efficient and Accurate Scene Text Detector. CVPR-2017
    """
    def __init__(self,
                 shrink_ratio=0.25,
                 ignore_ratio=0.6,
                 geometry='RBOX',
                 label_shape='Normal',
                 min_text_width=0,
                 max_text_width=2000,
                 lib_name=None,
                 lib_dir=None
                 ):
        """
        Args:
            shrink_ratio(float):  shrink polygon generation factor
            ignore_ratio(float): control pixel fractional ratio (calculated by foreground/background), used in training
            geometry(string): mode of data generation, in range of ["RBOX',"QUAD"]
            label_shape(string): mode of label genration，in range of ['Normal','Gaussian']
            min_text_width(int): allowed minimum text width, otherwise will be set as IGNORE
            max_text_width(int): allowed maximum text width, otherwise will be set as IGNORE
            lib_name(str): lib name of calling the function of ground-truth label generation
            lib_dir(str): lib path to calling the function of ground-truth label generation
        """
        if lib_name is None or not os.path.isfile(os.path.join(lib_dir, lib_name)):
            # Using default lib
            cur_path = os.path.realpath(__file__)
            lib_dir = cur_path.replace('\\', '/').split('/')[:-1]
            lib_dir = "/".join(lib_dir)+'/lib'
            lib_name = "east_data.so"
        self.shrink_ratio = shrink_ratio
        self.ignore_ratio = ignore_ratio
        self.geometry = geometry
        assert self.geometry in ['RBOX', 'QUAD'], "geometry only supports 'RBOX' and 'QUAD' mode!"
        self.label_shape = label_shape
        assert self.label_shape in ['Normal', 'Gaussian'], "label shape only supports 'Normal' and " \
                                                           "'Gaussian' mode!"
        self.min_text_width = min_text_width
        self.max_text_width = max_text_width

        if lib_name is not None and lib_dir is not None:
            lib = ctl.load_library(lib_name, lib_dir)
            self.generate_func = lib.parse_east_data
            self.generate_func.argtypes = [
                c_int,                                                     # image height
                c_int,                                                     # image width
                ctl.ndpointer(np.int32, ndim=2, flags='C_CONTIGUOUS'),  # gt_boxes
                c_int,                                                    # length of gt_boxes
                ctl.ndpointer(np.int32, ndim=2, flags='C_CONTIGUOUS'),  # gt_boxes_ignore
                c_int,                                                    # length of gt_boxes_ignore
                c_int,                                                    # pool_ratio
                c_int,                                                    # geometry mode
                c_int,                                                    # label shape
                c_float,                                                  # shrink_ratio
                c_float,                                                  # ignore_ratio
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),        # output gt_score_map
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),          # output weight of gt_score_map
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),        # output gt_geo_map
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),        # output weight of gt_geo_map
                c_int,                                                    # random seed
            ]
        else:
            self.generate_func = None

    def __call__(self, results):
        """ Main process of data generation.

        Args:
            results(dict): data flow

        Returns:
            dict: updated data flow
        """
        if self.generate_func is not None:
            gts = self._parse_east_data_cpp(results['pad_shape'], results['gt_poly_bboxes'],
                                            results['gt_poly_bboxes_ignore'], 4)
            gt_masks = np.concatenate([gts['gt_score_map'][np.newaxis, :, :],
                                       gts['gt_score_map_mask'][np.newaxis, :, :],
                                       gts['gt_geo_map'],
                                       gts['gt_geo_map_weight']],
                                      axis=0)
        else:
            raise NotImplementedError('generate_func is None!')
        results['gt_masks'] = gt_masks
        return results

    def _parse_east_data_cpp(self, img_shape, gt_boxes, gt_boxes_ignore, pool_ratio=4):
        """ Load C++ lib to generate ground-truth

        Args:
            img_shape(tuple): current image shape (pad_shape)
            gt_boxes(list[list[float]]): ground truth bboxes, in shape of [[x0, y0, x1, y1, ..., x3, y3], ...]
            gt_boxes_ignore(list[list[float]]): ignored ground truth bboxes,
                                                 in shape of [[x0, y0, x1, y1, ..., x3, y3], ...]
            pool_ratio(int): downsampling ratio of ground-truth map wrt original image

        Returns:
            dict:   All gts in a dict, including:
                gts['gt_score_map']: target score map
                gts['gt_score_map_mask']: target score map mask, 0 for ignore, 1 for not ignore
                gts['gt_geo_map']: target geo map
                gts['gt_geo_map_weight']: target geo map mask, 0 for ignore, 1 for not ignore
        """
        height, width, _ = img_shape
        height_new = int(height / pool_ratio)
        width_new = int(width / pool_ratio)

        # Filter out box whose length not equal to 8, and box whose edge length not support
        gt_boxes_valid = []
        gt_boxes_ignore_valid = []
        for box in gt_boxes:
            invalid = False
            if len(box) != 8:
                print("invalid annotation {}".format(box))
                continue
            for k in range(4):
                # filter out box with unspport text length
                if not (self.min_text_width <= math.sqrt((box[k * 2] - box[(k + 1) * 2 % 8]) ** 2 + (
                        box[k * 2 + 1] - box[(k + 1) * 2 % 8 + 1]) ** 2) <= self.max_text_width):
                    invalid = True
                    print("filter out oversized text")
                    gt_boxes_ignore_valid.append(box)
            if not invalid:
                gt_boxes_valid.append(np.array(box, dtype=np.int32))
        for box in gt_boxes_ignore:
            if len(box) != 8:
                continue
            gt_boxes_ignore_valid.append(np.array(box, dtype=np.int32))

        # Used to store output gts.
        gt_score_map = np.zeros(height_new * width_new, dtype=np.float32)
        gt_score_map_mask = np.ones(height_new * width_new, dtype=np.int32)
        seed = random.randint(0, 999999999)

        if self.geometry == 'RBOX':
            # In RBOX mode, geo data is a map with 5 channels, [d1, d2, d3, d4, theta]
            gt_geo_map = np.zeros(5 * height_new * width_new, dtype=np.float32)
            gt_geo_map_weight = np.zeros(5 * height_new * width_new, dtype=np.float32)
            geometry_mode = 0
        else:
            # In QUAD mode, geo data is a map with 5 channels, [dx_1, dy_1, dx_2, dy_2, dx_3, dy_3, dx_4, dy_4]
            gt_geo_map = np.zeros(8 * height_new * width_new, dtype=np.float32)
            gt_geo_map_weight = np.zeros(8 * height_new * width_new, dtype=np.float32)
            geometry_mode = 1

        if self.label_shape == 'Normal':
            label_shape_mode = 0
        else:
            label_shape_mode = 1

        # Calling C++ lib
        if gt_boxes_valid:
            gt_bboxes_np = np.array(gt_boxes_valid, dtype=np.int32)

        else:
            gt_bboxes_np = np.zeros((0, 8), dtype=np.int32)

        if gt_boxes_ignore_valid:
            gt_bboxes_ignore_np = np.array(gt_boxes_ignore_valid, dtype=np.int32)
        else:
            gt_bboxes_ignore_np = np.zeros((0, 8), dtype=np.int32)

        self.generate_func(height,
                           width,
                           gt_bboxes_np,
                           len(gt_boxes_valid),
                           gt_bboxes_ignore_np,
                           len(gt_boxes_ignore_valid),
                           pool_ratio,
                           geometry_mode,
                           label_shape_mode,
                           c_float(self.shrink_ratio),
                           c_float(self.ignore_ratio),
                           gt_score_map,
                           gt_score_map_mask,
                           gt_geo_map,
                           gt_geo_map_weight,
                           seed)

        gt_score_map = np.array(gt_score_map.reshape(height_new, width_new), dtype=np.float32)
        gt_score_map_mask = np.array(gt_score_map_mask.reshape(height_new, width_new), dtype=np.uint8)

        if self.geometry == 'RBOX':
            gt_geo_map = np.array(np.reshape(gt_geo_map, (5, height_new, width_new)), dtype=np.float32)
            gt_geo_map_weight = np.array(np.reshape(gt_geo_map_weight,(5, height_new, width_new)), dtype=np.float32)
        elif self.geometry == 'QUAD':
            gt_geo_map = np.array(np.reshape(gt_geo_map,(8, height_new, width_new)), dtype=np.float32)
            gt_geo_map_weight = np.array(np.reshape(gt_geo_map_weight,(8, height_new, width_new)), dtype=np.float32)

        gts = dict()
        gts['gt_score_map'] = gt_score_map
        gts['gt_score_map_mask'] = gt_score_map_mask
        gts['gt_geo_map'] = gt_geo_map
        gts['gt_geo_map_weight'] = gt_geo_map_weight

        return gts
