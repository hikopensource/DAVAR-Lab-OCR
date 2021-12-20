"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    post_mango.py
# Abstract       :    Post processing of MANGO text spotter.
                      Format the inference results of MANGO.

# Current Version:    1.0.0
# Date           :    2021-05-31
######################################################################################################
"""
import os
import math
from ctypes import c_int, c_float

import torch
import torch.nn.functional as F

import numpy as np
import numpy.ctypeslib as ctl

from davarocr.davar_common.core import POSTPROCESS
from .post_spotter_base import BasePostSpotter


@POSTPROCESS.register_module()
class PostMango(BasePostSpotter):
    """ Format the inference results of MANGO: (1) merge the predictions of grids
        (2) Scale bboxes into original image shape (3) Generate visualization mask
    """

    def __init__(self,
                 do_visualization=False,
                 seg_thr=0.5,
                 cate_thr=0.5,
                 lib_name=None,
                 lib_dir='./'
                 ):
        """
        Args:
            do_visualization (bool): Whether to generate 'seg_preds' and 'character_mask_att_preds' for visualization.
            seg_thr (float): Segmatation map threhold.
            cate_thr (float): Category map threhold.
            lib_name (str): C++ extend lib filename.
            lib_dir (str): C++ extend lib path.
        """

        super().__init__()

        # If there is no identified lib path, use the default path
        if lib_name is None or not os.path.isfile(os.path.join(lib_dir, lib_name)):
            cur_path = os.path.realpath(__file__)
            lib_dir = cur_path.replace('\\', '/').split('/')[:-1]
            lib_dir = "/".join(lib_dir) + '/lib'
            lib_name = "bfs_search.so"

        self.do_visualization = do_visualization
        self.seg_thr = seg_thr
        self.cate_thr = cate_thr

        assert 0.0 <= self.seg_thr <= 1.0
        assert 0.0 <= self.cate_thr <= 1.0

        lib = ctl.load_library(lib_name, lib_dir)
        self.generate_func = lib.generate_result
        self.generate_func.argtypes = [
            ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # seg_pred
            ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # cate_conf
            c_int,    # height
            c_int,    # width
            c_int,    # num_grid
            c_float,  # thres_seg
            c_float,  # thres_grid
            ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),  # seg_result
            ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),  # cate_result
            ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),  # cate_weight
            ctl.ndpointer(np.int32, flags='C_CONTIGUOUS'),  # seg_num
        ]

    def mask_reisze(self, mask_pred, img_meta):
        """ Convert predicted segmentation  & character attention mask to original image shape for visualization.

        Args:
            mask_preds (Tensor): forward output, in shape of [B, C, H, W]
            img_metas (dict): containing three keywords:
                              'ori_shape': original image shape
                              'img_shape': resized image shape
                              'pad_shape': padded image shape
        Returns:
            Tensor: mask predictions in original shapes. in shape of [B, C, H_new, W_new]
        """

        ori_shape = img_meta['ori_shape']
        img_shape = img_meta['img_shape']
        pad_shape = img_meta['pad_shape']

        pad_mask = F.interpolate(
            mask_pred, size=(pad_shape[0], pad_shape[1]), mode='bilinear')
        img_mask = pad_mask[:, :, :img_shape[0], :img_shape[1]]
        org_mask = F.interpolate(
            img_mask, size=(ori_shape[0], ori_shape[1]), mode='bilinear'
        )

        return org_mask.cpu().numpy()


    def post_processing(self, batch_result, img_metas, **kwargs):
        """ Post process of Mango (in single-level feature map only)

        Args:
            batch_result (dict): forward output needs to be formatted, including keys:
                                 'cate_preds', indicator of grid's category, Tensor(B, num_grid^2)
                                 'seg_preds', global segmentation, Tensor(B, 1, H, W)
                                 'bboxes_preds', polygon boxes that inferred from seg_preds, list(np.array(N, 8)), len = B
                                 'text_preds', predict transcriptions, list(["text1", "text2", ...],...[]), len = num_grid^2
                                 'character_mask_att_preds': cma visualization, Tensor(B, K, L, H, W)
            img_metas (dict): image meta info
            **kwargs: other parameters

        Returns:
            list(dict): Format results, like [{'points':[[x1, y1, ..., xn, yn],[],...], 'texts':["apple", "banana",...],
                        'seg_preds': np.ndarray, 'character_mask_att_preds': np.ndarray}, ...]
        """

        cate_preds = batch_result['cate_preds']
        seg_preds = batch_result['seg_preds']
        bboxes_preds = batch_result['bboxes_preds']
        cate_weights = batch_result['cate_weights']

        assert len(bboxes_preds) == seg_preds.shape[0] == cate_preds.shape[0]   # Batch size

        text_preds = batch_result['text_preds']

        character_mask_att_preds = batch_result['character_mask_att_preds']
        results = []
        for i in range(len(bboxes_preds)):
            result = dict()
            result['cate_preds'] = cate_preds[i].cpu().numpy()
            result['cate_weights'] = cate_weights[i]
            if self.do_visualization:
                result['seg_preds'] = self.mask_reisze(seg_preds, img_metas[i])[i]
            else:
                result['seg_preds'] = None
            if text_preds is None:
                result['points'] = []
                result['texts'] = []
                result['character_mask_att_preds'] = None
                results.append(result)
                break

            # The value in cate_preds indicates which instance the grid belongs to
            # Calculate the number of valid grid
            max_category_num = torch.max(torch.sum(torch.ge(cate_preds[i], 1), dim=0))

            # Calculate the location of valid grid
            _, cate_indices = torch.topk(cate_preds[i], int(max_category_num))

            # Collect the values of valid grid
            cate_preds_i = torch.gather(cate_preds[i], 0, cate_indices)
            cate_indices = cate_indices.cpu().numpy()

            # Collect the IOU of valid grid and its corresponding text instance
            cate_weights_i = cate_weights[i][cate_indices]
            final_text_preds = []
            final_bbox_preds = []
            scale_factor = img_metas[i]['scale_factor']
            if isinstance(scale_factor, (np.ndarray, list)):
                scale_factor = (scale_factor[0] + scale_factor[1]) / 2
                for j, scale_bbox in enumerate(bboxes_preds[i]):
                    # Scale bboxes into original image shape
                    scale_bbox = scale_bbox / scale_factor

                    # Calculate the valid grid corresponding to text instance
                    indices = torch.where(cate_preds_i == j + 1)[0]
                    if len(indices) == 0:
                        continue
                    voted_text = ""
                    text_dict = dict()
                    for ind in indices:
                        # Add end symbol to text
                        text = text_preds[ind] + "#"
                        weight = cate_weights_i[ind]
                        # Character voting according to weight
                        for index, char in enumerate(text):
                            if index not in text_dict:
                                text_dict[index] = dict()
                            if char in text_dict[index]:
                                text_dict[index][char] += weight
                            else:
                                text_dict[index][char] = weight

                    # Calculate the voting result for each character
                    for index in range(len(text_dict)):
                        char_dict = text_dict[index]
                        char_dict = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
                        if char_dict[0][0] == "#":
                            break
                        voted_text += char_dict[0][0]
                    final_text_preds.append(voted_text)
                    final_bbox_preds.append(scale_bbox.astype(np.int32).tolist())

            result['points'] = final_bbox_preds
            result['texts'] = final_text_preds
            if self.do_visualization:
                tmp_character_mask_att_preds, _ = torch.max(character_mask_att_preds, dim=1)
                tmp_character_mask_att_preds = torch.sigmoid(tmp_character_mask_att_preds)
                result['character_mask_att_preds'] = self.mask_reisze(tmp_character_mask_att_preds, img_metas[i])[i]
            else:
                result['character_mask_att_preds'] = None
            results.append(result)

        return results

    def bfs_search(self, seg_pred, cate_conf):
        """ c++ implement of BFS(Breadth-First-Search).

        Args:
            seg_pred (Tensor): global segmentation, Tensor(1, H, W)
            cate_conf (Tensor): confidence of grid's category, Tensor(num_grid^2)

        Returns:
            np.ndarray: bounding boxes predictions, in shape of [N, 8]
        Returns:
            Tensor: grid's category predictions, in shape of [1, num_grid^2]
        Returns:
            np.ndarray: weight of grid's category, in shape of [num_grid^2]
        """

        _, height, width = seg_pred.shape

        # Flatten
        seg_pred = seg_pred.cpu().numpy().reshape(-1)
        cate_conf = cate_conf.cpu().numpy().reshape(-1)
        num_grid = int(math.sqrt(len(cate_conf)))

        # Initialize variables
        seg_result = np.zeros(800, dtype=np.int32)
        cate_result = np.zeros(num_grid**2, dtype=np.int32)
        cate_weight = np.zeros(num_grid**2, dtype=np.int32)
        seg_num = np.zeros(1, dtype=np.int32)

        # Call bfs function
        self.generate_func(seg_pred, cate_conf, height, width, num_grid,
            self.seg_thr, self.cate_thr, seg_result, cate_result, cate_weight, seg_num)

        # Reweight cate_weight to interval [0, 1]
        cate_weight = cate_weight * num_grid ** 2 / height / width
        seg_num = seg_num[0]

        # Select the result according to the number of instances
        seg_result = seg_result.reshape(-1, 8)[:seg_num]
        cate_result = torch.IntTensor(cate_result).cuda()
        cate_result = cate_result.unsqueeze(0)
        return seg_result, cate_result, cate_weight
