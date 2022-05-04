"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    chargrid_layout.py
# Abstract       :    Text Detection model based on chargrid-net referred in paper
                      "Chargrid: Towards Understanding 2D Documents"

# Current Version:    1.0.0
# Date           :    2022-04-11
##################################################################################################
"""
import torch

from mmdet.models.builder import DETECTORS
from davarocr.davar_det.models.detectors import MaskRCNNDet


@DETECTORS.register_module()
class ChargridNetTextDet(MaskRCNNDet):
    def __init__(self,
                 use_chargrid=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_chargrid = use_chargrid

    def forward_train(self,
                      img,
                      img_metas,
                      chargrid_map,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        input_img = chargrid_map if self.use_chargrid else img
        return super().forward_train(img=input_img,
                                        img_metas=img_metas,
                                        gt_bboxes=gt_bboxes,
                                        gt_labels=gt_labels,
                                        gt_bboxes_ignore=gt_bboxes_ignore,
                                        gt_masks=gt_masks,
                                        proposals=proposals,
                                        **kwargs)

    def simple_test(self,
                    img,
                    img_metas,
                    chargrid_map,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        input_img = chargrid_map if self.use_chargrid else img
        if self.use_chargrid:
            input_img = torch.cat(input_img, dim=0)

        return super(ChargridNetTextDet, self).simple_test(img=input_img,
                                                           img_metas=img_metas,
                                                           proposals=proposals,
                                                           rescale=rescale,
                                                           **kwargs)
