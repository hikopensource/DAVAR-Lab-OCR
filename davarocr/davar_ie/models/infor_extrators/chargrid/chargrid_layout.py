"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    chargrid_layout.py
# Abstract       :    Layout analysis model based on chargrid-net referred in paper
                      "Chargrid: Towards Understanding 2D Documents"

# Current Version:    1.0.0
# Date           :    2022-04-12
##################################################################################################
"""
import torch

from mmdet.models.builder import DETECTORS
from davarocr.davar_det.models.detectors import MaskRCNNDet
from davarocr.davar_common.models.builder import build_embedding


@DETECTORS.register_module()
class ChargridNetLayout(MaskRCNNDet):
    def __init__(self,
                 embedding,
                 use_chargrid=True,
                 is_cat=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert (not is_cat) or use_chargrid
        self.use_chargrid = use_chargrid
        self.is_cat = is_cat
        self.embedding = build_embedding(embedding)

    def forward_train(self,
                      img,
                      img_metas,
                      chargrid_map,
                      gt_bboxes_2,
                      gt_labels_2,
                      gt_bboxes_ignore_2=None,
                      gt_masks_2=None,
                      proposals=None,
                      **kwargs):
        if not self.is_cat:
            input_img = chargrid_map if self.use_chargrid else img
        else:
            # chargrid_map = chargrid_map.sum(dim=1).unsqueeze(1).long()
            # chargrid_embedding = \
            #     self.embedding(chargrid_map).squeeze(1).permute(0, 3, 1, 2).contiguous()
            input_img = torch.cat([img, chargrid_map], dim=1)
            # input_img = torch.cat([img, chargrid_embedding], dim=1)

        return super().forward_train(img=input_img,
                                     img_metas=img_metas,
                                     gt_bboxes=gt_bboxes_2,
                                     gt_labels=gt_labels_2,
                                     gt_bboxes_ignore=gt_bboxes_ignore_2,
                                     gt_masks=gt_masks_2,
                                     proposals=proposals,
                                     **kwargs)

    def simple_test(self,
                    img,
                    img_metas,
                    chargrid_map,
                    proposals=None,
                    rescale=False):
        if self.use_chargrid:
            chargrid_map = torch.cat(chargrid_map, dim=0)
        if not self.is_cat:
            input_img = chargrid_map if self.use_chargrid else img
        else:
            # chargrid_map = chargrid_map.sum(dim=1).unsqueeze(1).long()
            # chargrid_embedding = \
            #     self.embedding(chargrid_map).squeeze(1).permute(0, 3, 1, 2).contiguous()
            input_img = torch.cat([img, chargrid_map], dim=1)
            # input_img = torch.cat([img, chargrid_embedding], dim=1)

        return super(MaskRCNNDet, self).simple_test(img=input_img,
                                                    img_metas=img_metas,
                                                    proposals=proposals,
                                                    rescale=rescale)
