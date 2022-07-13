"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_data.py
# Abstract       :    GT_mask generating in Text Perceptron

# Current Version:    1.0.0
# Date           :    2020-05-31
###################################################################################################
"""

from ctypes import c_int, c_float
import os
import numpy as np
import numpy.ctypeslib as ctl

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class TPDataGeneration:
    """Ground-Truth label generation in Text Perceptron model training, including segmentation and regression.[1]

    Ref: [1] Text Perceptron: Towards End-to-End Arbitrary Shaped Text Spotting. AAAI-20.
                <https://arxiv.org/abs/2002.06820>`_
    """

    def __init__(self,
                 shrink_head_ratio=0.25,
                 shrink_bond_ratio=0.1,
                 ignore_ratio=0.6,
                 lib_name=None,
                 lib_dir=None,
                 ):
        """ Ground-Truth label generation in Text Perceptron model training.

        Args:
            shrink_head_ratio(float):    scaling factor when generating head and tail boundaries
            shrink_bond_ratio(float):    scaling factor when generating top and bottom boundaries
            ignore_ratio(float)	    :    control pixel fractional ratio (calculated by foreground/background)
            lib_name(str)		    :    lib name of calling the function of ground-truth label generation
            lib_dir(str)		    :    lib path to calling the function of ground-truth label generation
        """

        # If there is no identified lib path, use the default path
        if lib_name is None or not os.path.isfile(os.path.join(lib_dir, lib_name)):
            cur_path = os.path.realpath(__file__)
            lib_dir = cur_path.replace('\\', '/').split('/')[:-1]
            lib_dir = "/".join(lib_dir)+'/lib'
            lib_name = "tp_data.so"

        self.shrink_head_ratio = shrink_head_ratio
        self.shrink_bond_ratio = shrink_bond_ratio
        self.ignore_ratio = ignore_ratio
        self.lib_name = lib_name
        self.lib_dir = lib_dir

        # The function of ground-truth label generation is implemented by C++ (inefficiency in Python),
        # and complied to .so file, called by ctypes due to computational
        if lib_name is not None and lib_dir is not None:
            # Lib loading
            lib = ctl.load_library(lib_name, lib_dir)
            self.generate_func = lib.parse_tp_data

            # Set types of function arguments
            self.generate_func.argtypes = [
                c_int,                                                         # height of image
                c_int,                                                         # width of image
                ctl.ndpointer(np.int32, ndim=2, flags='C_CONTIGUOUS'),       # gt_boxes
                c_int,                                                         # length of gt_boxes
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),               # length of each box in gt_boxes
                ctl.ndpointer(np.int32, ndim=2, flags='C_CONTIGUOUS'),       # gt_boxes_ignore
                c_int,                                                         # length of gt_boxes_ignore
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),               # lenght of each box in gt_bboxes_ignore
                c_int,                                                         # downsampling ratio
                c_float,                                                       # shrink_head_ratio
                c_float,                                                       # shrink_bond_ratio
                c_float,                                                       # ignore_ratio
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),               # target score_map
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),               # weight mask of target score_map
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),             # target geo_map_head
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),             # weight mask of target geo_map_head
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),             # target geo_map_tail
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),             # weight mask of target geo_map_tail
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),             # target geo_map_boundary
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),             # weight mask of target geo_map_boundary
            ]
        else:
            raise ValueError('lib_name or lib_dir cannot be None')

    def __call__(self, results):
        """Data generation pipeline

        Args:
            results(dict): Data flow, requires
                           results['pad_shape'],    image shape after padding tupe(3, H, W)
                           results['gt_poly_bboxes'], ground-truth poly boxes, list[[x1, y1, ...,xn,ym],...]
                           results['gt_poly_bboxes_ignore'],  ignored ground-truth poly boxes,
                                                              list[[x1, y1, ...,xn,ym],...]
        Returns:
            dict:  Data flow, updated
                   results['gt_masks]:  np.ndarray(N, 26, H, W] where
                                        gt_mask:[:,0]    :  gt_score_map
                                        gt_mask:[:,1]    :  gt_score_map_mask, 1 Care / 0 Not Care
                                        gt_mask:[:,2:6]  :  gt_geo_map_head
                                        gt_mask:[:,6:10] :  gt_geo_map_head_weight
                                        gt_mask:[:,10:14]:  gt_geo_map_tail
                                        gt_mask:[:,14:18]:  gt_geo_map_tail_weight
                                        gt_mask:[:,18:22]:  gt_geo_map_bond
                                        gt_mask:[:,22:26]:  gt_geo_map_bond_weight

        """
        gts = self._parse_tp_data_cpp(results['pad_shape'], results['gt_poly_bboxes'],
                                      results['gt_poly_bboxes_ignore'], 4)

        # Concatenate all the ground_truth lable
        gt_masks = np.concatenate(
            [gts['gt_score_map'][np.newaxis, :, :], gts['gt_score_map_mask'][np.newaxis, :, :],
             gts['gt_geo_map_head'], gts['gt_geo_map_head_weight'], gts['gt_geo_map_tail'],
             gts['gt_geo_map_tail_weight'], gts['gt_geo_map_bond'], gts['gt_geo_map_bond_weight']], axis=0)
        results['gt_masks'] = gt_masks
        return results

    def _parse_tp_data_cpp(self, img_shape, gt_boxes, gt_boxes_ignore, pool_ratio=4):
        """Parsing and generating gt_mask for training, by calling C++ lib

        Args:
            img_shape(Tuple)                  :  image size (pad_shape)
            gt_boxes(list[list[float]]        :  detection ground-truth boxes [[x1, y1, x2, y2, ..., xn, yn], ...]
            gt_bboxes_ignore(list[list[float]]:  ignored detection ground-truth boxes [[x1, y2, x2, y2, ....],...[...]]
            pool_ratio(int)                   :  downsampling ratio of ground-truth map wrt original image

        Returns:
            dict:   All gts in a dict, including
                    gt_score_map            :    target segmentation ground-truth [H x W]
                    gt_score_map_mask       :    weight mask of target segmentation map （ignored if 0）[H x W]
                    gt_geo_map_head         :    pixel regression ground-truth map of target head boundary area
                                                 [4 x H x W]
                    gt_score_map_head_weight:    weight mask of pixel regression ground-truth map of target head
                                                 boundary area [4 x H x W]
                    gt_geo_map_tail         :    pixel regression ground-truth map of target tail boundary area
                                                 [4 x H x W]
                    gt_score_map_tail_weight:    weight mask of pixel regression ground-truth map of target tail
                                                 boudary area [4 x H x W]
                    gt_geo_map_bond         :    pixel regression ground-truth map of target center area [4 x H x W]
                    gt_score_map_bond_weight:    weight mask of pixel regression ground-truth map of target center
                                                 area [4 x H x W]
        """

        height, width, _ = img_shape
        new_height = int(height / pool_ratio)
        new_width = int(width / pool_ratio)

        gt_boxes_np = np.zeros((len(gt_boxes), 48), dtype=np.int32)
        gt_boxes_length_np = np.ones(len(gt_boxes), dtype=np.int32)
        gt_boxes_ignore_np = np.zeros((len(gt_boxes_ignore), 48), dtype=np.int32)
        gt_boxes_ignore_length_np = np.ones(len(gt_boxes_ignore), dtype=np.int32)

        # Filter out boxes with length greater than 24 points
        for i, box in enumerate(gt_boxes):
            if len(box) > 48:
                print("Point length larger than 48!")
            gt_boxes_length_np[i] = len(box)
            for j, box_j in enumerate(box):
                gt_boxes_np[i, j] = box_j

        for i, box in enumerate(gt_boxes_ignore):
            if len(box) > 48:
                print("Point length larger than 48!")
            gt_boxes_ignore_length_np[i] = len(box)
            for j, box_j in enumerate(box):
                gt_boxes_ignore_np[i, j] = box_j

        # Allocate spaces for returned results
        gt_score_map = np.zeros(new_height * new_width, dtype=np.int32)
        gt_mask = np.ones(new_height * new_width, dtype=np.int32)
        gt_geo_head = np.zeros(4 * new_height * new_width, dtype=np.float32)
        gt_geo_head_weight = np.zeros(4 * new_height * new_width,dtype=np.float32)
        gt_geo_tail = np.zeros(4 * new_height * new_width, dtype=np.float32)
        gt_geo_tail_weight = np.zeros(4 * new_height * new_width,dtype=np.float32)
        gt_geo_bond = np.zeros(4 * new_height * new_width, dtype=np.float32)
        gt_geo_bond_weight = np.zeros(4 * new_height * new_width,dtype=np.float32)

        # Calling C++ lib
        self.generate_func(height, width, gt_boxes_np, len(gt_boxes), gt_boxes_length_np, gt_boxes_ignore_np,
                           len(gt_boxes_ignore), gt_boxes_ignore_length_np, pool_ratio, c_float(self.shrink_head_ratio),
                           c_float(self.shrink_bond_ratio), c_float(self.ignore_ratio), gt_score_map, gt_mask,
                           gt_geo_head, gt_geo_head_weight, gt_geo_tail, gt_geo_tail_weight, gt_geo_bond,
                           gt_geo_bond_weight)

        # Return formatted results
        gts = dict()
        gts['gt_score_map'] = np.array(gt_score_map.reshape(new_height, new_width), dtype=np.uint8)
        gts['gt_score_map_mask'] = np.array(gt_mask.reshape(new_height, new_width), dtype=np.uint8)
        gts['gt_geo_map_head'] = np.array(np.reshape(gt_geo_head, (4, new_height, new_width)), dtype=np.float32)
        np.reshape(gt_geo_head, (4, new_height, new_width))
        gts['gt_geo_map_head_weight'] = np.array(np.reshape(gt_geo_head_weight, (4, new_height, new_width)),
                                                 dtype=np.float32)
        gts['gt_geo_map_tail'] = np.array(np.reshape(gt_geo_tail, (4, new_height, new_width)), dtype=np.float32)
        gts['gt_geo_map_tail_weight'] = np.array(np.reshape(gt_geo_tail_weight, (4, new_height, new_width)),
                                                 dtype=np.float32)
        gts['gt_geo_map_bond'] = np.array(np.reshape(gt_geo_bond, (4, new_height, new_width)), dtype=np.float32)
        gts['gt_geo_map_bond_weight'] = np.array(np.reshape(gt_geo_bond_weight, (4, new_height, new_width)),
                                                 dtype=np.float32)

        return gts
