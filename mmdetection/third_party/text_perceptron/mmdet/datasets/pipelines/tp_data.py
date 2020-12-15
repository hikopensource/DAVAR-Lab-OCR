"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tp_data.py
# Abstract       :    GT_mask generating in Text Perceptron

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31

# Modified Date  :    2020-11-25
# Modified by    :    inusheng
# Comments       :    Code and comment standardized
###################################################################################################
"""

from ctypes import c_int, c_float
import numpy as np
import numpy.ctypeslib as ctl

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.registry import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module
class TPDataGeneration():
    """
    Description:
        Ground-Truth label generation in Text Perceptron model training,
        including segmentation and regression.
        
    Properties:
        shrink_head_ratio (float):    scaling factor when generating head and tail boundaries
        shrink_bond_ratio (float):    scaling factor when generating top and bottom boundaries
        ignore_ratio (float)	 :    control pixel fractional ratio (calculated by foreground/background), used in training
        lib_name(str)		 :    lib name of calling the function of ground-truth label generation
        lib_dir(str)		 :    lib path to calling the function of ground-truth label generation
    """

    def __init__(self,
                 shrink_head_ratio=0.25,
                 shrink_bond_ratio=0.1,
                 ignore_ratio=0.6,
                 lib_name=None,
                 lib_dir=None,
                 ):

        self.shrink_head_ratio = shrink_head_ratio
        self.shrink_bond_ratio = shrink_bond_ratio
        self.ignore_ratio = ignore_ratio
        self.lib_name = lib_name
        self.lib_dir = lib_dir

        # The function of ground-truth label generation is implemented by C++,
        # and complied to .so file, called by ctypes due to computational
        # inefficiency in Python
        if lib_name is not None and lib_dir is not None:
            # lib loading
            lib = ctl.load_library(lib_name, lib_dir)
            self.generate_func = lib.parse_tp_data

            # set types of function arguments
            self.generate_func.argtypes = [
                c_int,                                                              # height of image
                c_int,                                                              # width of image
                ctl.ndpointer(np.int32, ndim=2, flags='C_CONTIGUOUS'),              # gt_boxes
                c_int,                                                              # length of gt_boxes
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),                      # length of each box in gt_boxes
                ctl.ndpointer(np.int32, ndim=2, flags='C_CONTIGUOUS'),              # gt_boxes_ignore
                c_int,                                                              # length of gt_boxes_ignore
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),                      # lenght of each box in gt_bboxes_ignore
                c_int,                                                              # downsampling ratio of feature maps wrt original image
                c_float,                                                            # shrink_head_ratio
                c_float,                                                            # shrink_bond_ratio
                c_float,                                                            # ignore_ratio
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),                      # target score_map
                ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),                      # weight mask of target score_map
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),                    # target geo_map_head
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),                    # weight mask of target geo_map_head
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),                    # target geo_map_tail
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),                    # weight mask of target geo_map_tail
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),                    # target geo_map_boundary
                ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),                    # weight mask of target geo_map_boundary
            ]
        else:
            raise ValueError('lib_name or lib_dir cannot be None')

        self.idx = 0  # debug only

    def __call__(self, results):
        gt_score_map, gt_score_map_mask, gt_geo_map_head, gt_geo_map_head_weight, gt_geo_map_tail, gt_geo_map_tail_weight, gt_geo_map_bond, gt_geo_map_bond_weight = self._parse_tp_data_cpp(
            results['pad_shape'], results['gt_bboxes'], results['gt_bboxes_ignore'], 4)

        # concatenate all the ground_truth lable
        gt_masks = np.concatenate([gt_score_map[np.newaxis, :, :], gt_score_map_mask[np.newaxis, :, :], gt_geo_map_head,
                                   gt_geo_map_head_weight, gt_geo_map_tail, gt_geo_map_tail_weight, gt_geo_map_bond,
                                   gt_geo_map_bond_weight], axis=0)
        results['gt_masks'] = gt_masks
        return results

    def _parse_tp_data_cpp(self, img_shape, gt_boxes, gt_boxes_ignore, pool_ratio=4):
        """
        Description:
            Parsing and generating gt_mask for training, by calling C++ lib
            
        Args:
            img_shape       :    image size (pad_shape)
            gt_boxes        :    detection ground-truth boxes [[x1, y1, x2, y2, ..., xn, yn], ...]
            gt_bboxes_ignore:    ignored detection ground-truth boxes [[x1, y2, x2, y2, ....],...[...]]
            pool_ratio      :    downsampling ratio of ground-truth map wrt original image
            
        Returns:
            gt_score_map            :    target segmentation ground-truth [H x W]
            gt_score_map_mask       :    weight mask of target segmentation map （ignored if 0）[H x W]
            gt_geo_map_head         :    pixel regression ground-truth map of target head boundary area [4 x H x W]
            gt_score_map_head_weight:    weight mask of pixel regression ground-truth map of target head boundary area [4 x H x W]
            gt_geo_map_tail         :    pixel regression ground-truth map of target tail boundary area [4 x H x W]
            gt_score_map_tail_weight:    weight mask of pixel regression ground-truth map of target tail boudary area [4 x H x W]
            gt_geo_map_bond         :    pixel regression ground-truth map of target center area [4 x H x W]
            gt_score_map_bond_weight:    weight mask of pixel regression ground-truth map of target center area [4 x H x W]
        """
        
        height, width, _ = img_shape
        new_height = int(height / pool_ratio)
        new_width = int(width / pool_ratio)

        gt_boxes_np = np.zeros((len(gt_boxes), 48), dtype=np.int32)
        gt_boxes_length_np = np.ones(len(gt_boxes), dtype=np.int32)
        gt_boxes_ignore_np = np.zeros((len(gt_boxes_ignore), 48), dtype=np.int32)
        gt_boxes_ignore_length_np = np.ones(len(gt_boxes_ignore), dtype=np.int32)
        
        # filter out boxes with length greater than 24 points
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

        # allocate spaces for returned results
        gt_score_map = np.zeros(new_height * new_width, dtype=np.int32)
        gt_mask = np.ones(new_height * new_width, dtype=np.int32)
        gt_geo_head = np.zeros(4 * new_height * new_width, dtype=np.float32)
        gt_geo_head_weight = np.zeros(4 * new_height * new_width,dtype=np.float32)
        gt_geo_tail = np.zeros(4 * new_height * new_width, dtype=np.float32)
        gt_geo_tail_weight = np.zeros(4 * new_height * new_width,dtype=np.float32)
        gt_geo_bond = np.zeros(4 * new_height * new_width, dtype=np.float32)
        gt_geo_bond_weight = np.zeros(4 * new_height * new_width,dtype=np.float32)

        # calling and excuting C++ lib
        self.generate_func(height, width, gt_boxes_np, len(gt_boxes), gt_boxes_length_np, gt_boxes_ignore_np,
                           len(gt_boxes_ignore), gt_boxes_ignore_length_np, pool_ratio, c_float(self.shrink_head_ratio),
                           c_float(self.shrink_bond_ratio), c_float(self.ignore_ratio), gt_score_map, gt_mask,
                           gt_geo_head, gt_geo_head_weight, gt_geo_tail, gt_geo_tail_weight, gt_geo_bond,
                           gt_geo_bond_weight)

        # return formatted results
        gt_score_map = np.array(gt_score_map.reshape(new_height, new_width), dtype=np.uint8)
        gt_mask = np.array(gt_mask.reshape(new_height, new_width), dtype=np.uint8)
        gt_geo_head = np.array(gt_geo_head.reshape(4, new_height, new_width), dtype=np.float32)
        gt_geo_head_weight = np.array(gt_geo_head_weight.reshape(4, new_height, new_width), dtype=np.float32)
        gt_geo_tail = np.array(gt_geo_tail.reshape(4, new_height, new_width), dtype=np.float32)
        gt_geo_tail_weight = np.array(gt_geo_tail_weight.reshape(4, new_height, new_width), dtype=np.float32)
        gt_geo_bond = np.array(gt_geo_bond.reshape(4, new_height, new_width), dtype=np.float32)
        gt_geo_bond_weight = np.array(gt_geo_bond_weight.reshape(4, new_height, new_width), dtype=np.float32)

        return gt_score_map, gt_mask, gt_geo_head, gt_geo_head_weight, gt_geo_tail, gt_geo_tail_weight, gt_geo_bond, gt_geo_bond_weight


@PIPELINES.register_module
class TPFormatBundle():
    """
    Description:
        format returned results, and convert gt_masks to tensor and store in Data Container
        img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
        gt_masks: (1)to tensor, (2)to DataContainer (stack=True)
    """

    def __call__(self, results):
        if 'img' in results:
            img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img).float(), stack=True)
        if 'gt_masks' in results:
            results['gt_masks'] = DC(to_tensor(results['gt_masks']), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__
