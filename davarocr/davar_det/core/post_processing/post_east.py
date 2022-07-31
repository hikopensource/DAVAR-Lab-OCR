"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    post_east.py
# Abstract       :    Post processing of EAST text detector.

# Current Version:    1.0.0
# Date           :    2020-05-31
######################################################################################################
"""
import os
import multiprocessing
from ctypes import c_int, c_float, POINTER
import numpy as np
import numpy.ctypeslib as ctl

from davarocr.davar_common.core import POSTPROCESS
from .post_detector_base import BasePostDetector


@POSTPROCESS.register_module()
class PostEAST(BasePostDetector):
    """ EAST post-processing, using geo map and score map to generate final boxes."""
    def __init__(self,
                 thres_text=0.8,
                 nms_thres=0.2,
                 nms_method="RBOX",
                 num_workers=0,
                 lib_name=None,
                 lib_dir="./",
                 ):
        """
        Args:
            thres_text(float): threshold to detect text
            nms_thres(float): iou threshold when conduct nms
            nms_method(str): nms mode, support for 'RBOX' and 'QUAD'.
            num_workers(int): process number, default for single process
        """
        super().__init__()
        self.thres_text = thres_text
        self.nms_thres = nms_thres
        assert nms_method in ['RBOX', 'QUAD'], "Only support nms_method in 'RBOX' and 'QUAD' mode "
        if nms_method == 'RBOX':
            self.nms_method = 0
        else:
            self.nms_method = 1
        self.count = 0
        assert 0.0 <= self.thres_text <= 1.0
        self.num_workers = num_workers
        if num_workers > 0:
            self.pool = multiprocessing.Pool(num_workers)
        self.count = 1

        if lib_name is None or not os.path.isfile(os.path.join(lib_dir, lib_name)):
            cur_path = os.path.realpath(__file__)
            lib_dir = cur_path.replace('\\', '/').split('/')[:-1]
            lib_dir = "/".join(lib_dir) + '/lib'
            lib_name = "east_postprocess.so"

        lib = ctl.load_library(lib_name, lib_dir)
        self.generate_func = lib.generate_result
        self.generate_func.argtypes = [
            ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # score map
            ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # geo map
            c_int,  # height
            c_int,  # width
            c_int,  # pool ratio
            c_float,  # scale factor
            c_float,  # thres_text
            c_float,  # nms_thres
            c_int,  # nms_method
            ctl.ndpointer(np.float32, flags='C_CONTIGUOUS'),  # result
            POINTER(c_int)  # result_num
            ]

    def post_east(self,
                  cur_score_map,
                  cur_geo_map,
                  height,
                  width,
                  pool_ratio,
                  scale_factor,
                  thres_text,
                  nms_thres,
                  nms_method,
                  ):
        """ Calling ctl lib to do post processing

        Args:
            cur_score_map(np.ndarray): predict score map, in shape of 1D with length of [Bx1xHxW]
            cur_geo_map(np.ndarray): predict geo map, in shape 1D with length of [Bx5xHxW] or [Bx8xHxW]
            height(int): feature map height, H
            width(int): feature map width, W
            pool_ratio(int): feature map ratio, defaltu as 4
            scale_factor(float):  ratio of original feature map to original image
            thres_text(float): threshold to detect text
            nms_thres(float): iou threshold when conduct nms
            nms_method(int): nms mode, 0: standard RBOX, 1: long-text RBOX, 2: standard QUAD, 2: long-text QUAD.
                             long-text mode means pixels only predict their nearest corner.

        Returns:
            np.ndarray: predict results, in shape of [N, 9], box coordinates and confidence.
        Returns:
            int: result counts.
        """

        result = np.zeros((width * height, 9),
                          dtype=np.float32)  # 8 for coordinates and 1 for confidence
        result_num = c_int()

        self.generate_func(cur_score_map.reshape(-1), cur_geo_map.reshape(-1),
                      height, width, pool_ratio, scale_factor,
                      thres_text, nms_thres, nms_method, result, result_num)
        return result, result_num.value


    def post_processing(self, results, img_meta):
        """
        Args:
            results: predict results.
            img_meta: meta information

        Returns:
            list(dict): format results in a dict
                     [{"points":[[x1,y1, x2, y2, ..., x4, y4],...], "confidence": [0.8,...] },...]
        """
        score_map, geo_map = results
        height = score_map.shape[2]
        width = score_map.shape[3]

        score_map = score_map.cpu().numpy()
        geo_map = geo_map.cpu().numpy()
        results_list = []

        res_list = []
        if self.num_workers > 0 and score_map.shape[0] > 1:
            # Multiprocessing
            for i in range(score_map.shape[0]):
                cur_score_map = score_map[i]
                cur_geo_map = geo_map[i]
                scale_factor = 1.0
                if 'scale_factor' in img_meta[i]:
                    if len(img_meta[i]['scale_factor']) > 1:
                        scale_factor = (img_meta[i]['scale_factor'][0] + img_meta[i]['scale_factor'][1]) / 2
                    else:
                        scale_factor = float(img_meta[i]['scale_factor'])

                res = self.post_east(cur_score_map.reshape(-1),
                                     cur_geo_map.reshape(-1),
                                     height, width, 4,
                                     scale_factor,
                                     self.thres_text,
                                     self.nms_thres,
                                     self.nms_method)
                res_list.append(res)
            res_list = [r.get() for r in res_list]
        else:
            # Single-processing
            for i in range(score_map.shape[0]):
                cur_score_map = score_map[i]
                cur_geo_map = geo_map[i]
                scale_factor = 1.0
                if 'scale_factor' in img_meta[i]:
                    if len(img_meta[i]['scale_factor']) > 1:
                        scale_factor = (img_meta[i]['scale_factor'][0] + img_meta[i]['scale_factor'][1]) / 2
                    else:
                        scale_factor = float(img_meta[i]['scale_factor'])

                res = self.post_east(cur_score_map.reshape(-1),
                                cur_geo_map.reshape(-1),
                                height, width, 4,
                                scale_factor,
                                self.thres_text,
                                self.nms_thres,
                                self.nms_method
                                )
                res_list.append(res)

        # Pack output into standard form
        for res in res_list:
            result, result_num = res
            results = dict()
            results["points"] = []
            results["confidence"] = []
            for i in range(result_num):
                points = result[i]
                results["points"].append(points[:8])
                results["confidence"].append(points[8])

            results_list.append(results)

        return results_list
