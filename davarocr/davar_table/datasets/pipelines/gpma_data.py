"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    gpma_data.py
# Abstract       :    Generating global pyramid mask of table (used in GPMA branch)

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

import os
from ctypes import c_int, c_float
import numpy as np
from numpy import ctypeslib as ctl
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GPMADataGeneration:
    """ Generate gt_mask for training(GPMA branch).

    Ref: Qiao L, Li Z, Cheng Z, et al. LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask
     Alignment[J]. arXiv preprint arXiv:2105.06224, 2021. (Accepted by ICDAR 2021, Best Industry Paper)

    """

    def __init__(self,
                 ignore_ratio=0.6,
                 shrink_ratio=(0, 1 / 10),
                 lib_name=None,
                 lib_dir=None
                 ):
        """Generate gt_mask for training(GPMA branch)

        Args:
            ignore_ratio (float): Controls the ratio of background pixels to foreground pixels
            shrink_ratio (tuple): Controls the ratio of shrink (ensure that virtual or real table lines are preserved)
            lib_name (str): lib name of calling the function of ground-truth label generation
            lib_dir (str): lib path to calling the function of ground-truth label generation
        """

        if lib_name is None or not os.path.isfile(os.path.join(lib_dir, lib_name)):
            # Using default lib
            cur_path = os.path.realpath(__file__)
            lib_dir = cur_path.replace('\\', '/').split('/')[:-1]
            lib_dir = "/".join(lib_dir) + '/lib'
            lib_name = "gpma_data.so"
        if lib_name is not None and lib_dir is not None:
            lib = ctl.load_library(lib_name, lib_dir)
            self.generate_func = lib.generate_seg_data
            self.generate_func.argtypes = [
                c_int,  # height
                c_int,  # width
                ctl.ndpointer(np.float32, ndim=2, flags='C_CONTIGUOUS'),  # gt_boxes
                c_int,  # gt_boxes_size
                ctl.ndpointer(np.float32, ndim=2, flags='C_CONTIGUOUS'),  # shrink_gt_boxes
                c_int,  # empty_gt_boxes_size
                ctl.ndpointer(np.float32, ndim=2, flags='C_CONTIGUOUS'),  # empty_gt_boxes
                c_int,  # empty_gt_boxes_size
                ctl.ndpointer(np.float32, ndim=2, flags='C_CONTIGUOUS'),  # small_gt_boxes
                c_int,  # small_gt_boxes_size
                c_int,  # pool_ratio,
                c_float,  # ignore_ratio
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),  # gt_score_map
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),  # gt_mask
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # gt_geo_data
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # gt_geo_weight
            ]
        else:
            self.generate_func = None
        self.ignore_ratio = ignore_ratio
        self.shrink_ratio = shrink_ratio

    def __call__(self, results):
        """Data generation pipeline

        Args:
            results(dict): Data flow, requires
                           results['pad_shape'], image shape after padding tupe(3, H, W)
                           results['gt_bboxes'], ground-truth bboxes of non-empty aligned cells [[x1, y1, x2, y2], ...]
                           results['gt_content_bboxes'], ground-truth bboxes of text regions [[x1, y2, x2, y2],...[...]]
                           results['gt_empty_bboxes'],ground-truth bboxes of empty aligned cells [[x1, y1, x2, y2], ...]

        Returns:
            results(dict):  Data flow, updated results['gt_semantic_seg]: np.ndarray(N, 6, H, W], where N is batch size
                                gt_semantic_seg:[:,0]: gt_cell_region
                                gt_semantic_seg:[:,1]: cell_region_weight, 1 Care / 0 Not Care
                                gt_semantic_seg:[:,2:4]: gt_global_pyramid
                                gt_semantic_seg:[:,4:6]: global_pyramid_weight, 1 Care / 0 Not Care

        """

        if 'gt_labels' not in results:
            results['gt_labels'] = [1] * len(results['gt_bboxes'])
        # shrinked bboxes of empty cells
        gt_empty_bboxes = []
        for bbox in results.get("gt_empty_bboxes", []):
            xoffset, yoffset = (bbox[2] - bbox[0]) * self.shrink_ratio[0], (bbox[3] - bbox[1]) * self.shrink_ratio[1]
            temp = [bbox[0] + xoffset, bbox[1] + yoffset, bbox[2] - xoffset, bbox[3] - yoffset]
            gt_empty_bboxes.append(temp)

        gt_semantic_seg = self._parse_gpma_data_cpp(results['pad_shape'], results['gt_bboxes'],
                                                    results['gt_content_bboxes'], gt_empty_bboxes)

        results['gt_semantic_seg'] = gt_semantic_seg

        return results

    def _parse_gpma_data_cpp(self, img_shape, gt_cell_bboxes, gt_content_bboxes, gt_empty_bboxes, pool_ratio=4):
        """Parsing and generating gt_mask for training(GPMA branch), by calling C++ lib

        Args:
            img_shape(Tuple): image size (pad_shape)
            gt_cell_bboxes(list[list[float]]): ground-truth bboxes of non-empty aligned cells [[x1, y1, x2, y2], ...]
            gt_content_bboxes(list[list[float]]): ground-truth bboxes of text regions [[x1, y2, x2, y2],...[...]]
            gt_empty_bboxes(list[list[float]]): ground-truth bboxes of empty aligned cells [[x1, y1, x2, y2], ...]
            pool_ratio(int): downsampling ratio of ground-truth map wrt original image

        Returns:
            np.array: All gts in a np.array, including
                    gt_cell_region: target aligned cell region mask ground-truth [H x W]
                    cell_region_weight: weight mask of target aligned cell region (ignored if 0) [H x W]
                    gt_global_pyramid: target global pyramid mask ground-truth [2 x H x W]
                    global_pyramid_weight: weight mask of target global pyramid mask (ignored if 0) [2 x H x W]
        """

        gt_shrink_bboxes = []
        for bbox in gt_cell_bboxes:
            xoffset, yoffset = (bbox[2] - bbox[0]) * self.shrink_ratio[0], (bbox[3] - bbox[1]) * self.shrink_ratio[1]
            temp = [bbox[0] + xoffset, bbox[1] + yoffset, bbox[2] - xoffset, bbox[3] - yoffset]
            gt_shrink_bboxes.append(temp)
        assert len(gt_cell_bboxes) == len(gt_content_bboxes) == len(gt_shrink_bboxes)

        gt_cell_bboxes = np.array([[box[0], box[1], box[2], box[3]] for box in gt_cell_bboxes], dtype=np.float32)
        gt_shrink_bboxes = np.array([[box[0], box[1], box[2], box[3]] for box in gt_shrink_bboxes], dtype=np.float32)
        gt_content_bboxes = np.array([[box[0], box[1], box[2], box[3]] for box in gt_content_bboxes], dtype=np.float32)
        if len(gt_empty_bboxes) == 0:
            gt_empty_bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_empty_bboxes = np.array([[box[0], box[1], box[2], box[3]] for box in gt_empty_bboxes], dtype=np.float32)

        height, width, _ = img_shape
        new_height = int(height / pool_ratio)
        new_width = int(width / pool_ratio)

        # Used to store output results.
        gt_cell_region = np.zeros(new_height * new_width, dtype=np.int32)
        cell_region_weight = np.ones(new_height * new_width, dtype=np.int32)
        gt_global_pyramid = np.zeros(2 * new_height * new_width, dtype=np.float32)
        global_pyramid_weight = np.zeros(2 * new_height * new_width, dtype=np.float32)

        # Call C++ lib execution
        self.generate_func(height, width, gt_cell_bboxes, len(gt_cell_bboxes), gt_shrink_bboxes, len(gt_shrink_bboxes),
                           gt_empty_bboxes, len(gt_empty_bboxes),
                           gt_content_bboxes, len(gt_content_bboxes),
                           pool_ratio, self.ignore_ratio, gt_cell_region, cell_region_weight,
                           gt_global_pyramid, global_pyramid_weight)

        # Return the formatted results
        gt_cell_region = np.array(gt_cell_region.reshape(new_height, new_width), dtype=np.int32)
        cell_region_weight = np.array(cell_region_weight.reshape(new_height, new_width), dtype=np.int32)
        gt_global_pyramid = np.array(gt_global_pyramid.reshape((2, new_height, new_width)), dtype=np.float32)
        global_pyramid_weight = np.array(global_pyramid_weight.reshape((2, new_height, new_width)), dtype=np.float32)

        return np.concatenate([gt_cell_region[np.newaxis, :, :], cell_region_weight[np.newaxis, :, :],
                               gt_global_pyramid, global_pyramid_weight], axis=0)
