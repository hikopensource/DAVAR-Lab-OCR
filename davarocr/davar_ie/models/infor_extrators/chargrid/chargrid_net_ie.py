"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    chargrid_net_ie.py
# Abstract       :    Information extract model based on chargrid-net referred in paper
                      "Chargrid: Towards Understanding 2D Documents"

# Current Version:    1.0.0
# Date           :    2022-03-22
##################################################################################################
"""
import torch
import torch.nn as nn
from davarocr.davar_spotting.models import SPOTTER
from mmdet.models.builder import build_roi_extractor, build_head
from mmdet.core import bbox2roi
from davarocr.davar_det.models.detectors import MaskRCNNDet


@SPOTTER.register_module()
class ChargridNetIE(MaskRCNNDet):
    def __init__(self,
                 use_chargrid=True,
                 ie_roi_extractor=None,
                 ie_cls_head=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_chargrid = use_chargrid
        self.ie_roi_extractor = build_roi_extractor(ie_roi_extractor)
        self.ie_cls_head = build_head(ie_cls_head)

    def process_infor_feats(self, infor_feats, gt_bboxes, gt_labels=None):
        """ infor features processing before being sent into classification head

            Args:
                infor_feats(Tensor): in shape of N x C x H x W, where N is all text bboxes in a batch
                gt_bboxes (list(Tensor)): box Tensor for each sample in a batch, e.g. [N_B x 4] x B
                gt_labels (list(Tensor)): category labels for all text bboxes, in shape of [N_B x 1] x B

            Returns:
                feat(Tensor): output feature, in [B x Max_N x C]
            Returns:
                batched_label(list(Tensor)): category label Tensor to same length, in [Max_N x 1] x B
        """
        pooling = nn.AdaptiveAvgPool2d((1, 1))
        max_length = max([per_b.size(0) for per_b in gt_bboxes])
        batch_size = len(gt_bboxes)

        batched_feat = []
        batched_label = []
        last_idx = 0

        for i in range(batch_size):
            b_s = gt_bboxes[i].size(0)
            feat_size = list(infor_feats.size())
            feat_size[0] = max_length - b_s
            # features
            batched_feat.append(
                torch.cat((infor_feats[last_idx: last_idx + b_s], infor_feats.new_full(feat_size, 0)), 0))
            # labels
            if gt_labels is not None:
                per_label = gt_labels[i]
                label_size = list(per_label.size())
                label_size[0] = max_length - b_s
                batched_label.append(
                    torch.cat((per_label, per_label.new_full(label_size, 255)), 0))

            last_idx += b_s
        # merge features of each sample into one feature
        feat = torch.cat(batched_feat, 0)   # [(B x Max_N) x C x H x W]
        feat = pooling(feat).squeeze(2).squeeze(2)   # [(B x Max_N) x C]
        feat = feat.view(batch_size, -1, feat.size(-1))    # [B x Max_N x C]

        return feat, batched_label

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
        losses =  super().forward_train(img=input_img,
                                        img_metas=img_metas,
                                        gt_bboxes=gt_bboxes,
                                        gt_labels=gt_labels,
                                        gt_bboxes_ignore=gt_bboxes_ignore,
                                        gt_masks=gt_masks,
                                        proposals=proposals,
                                        **kwargs)
        # text recognition branch
        if sum([box.shape[0] for box in gt_bboxes]) == 0:
            # non positive gts in current batch, we just not train the recognition branch
            return losses

        img_feat = self.extract_feat(input_img)
        recog_rois = bbox2roi(gt_bboxes)
        infor_feats = self.ie_roi_extractor(img_feat[:self.ie_roi_extractor.num_inputs], recog_rois)
        feat, batched_label = self.process_infor_feats(infor_feats, gt_bboxes, gt_labels)

        # node classification task
        cls_pred = self.ie_cls_head(feat)
        cls_pred = cls_pred.view(-1, cls_pred.size(-1))
        tmp_labels = torch.cat(batched_label, 0).view(-1)
        valid_mask = tmp_labels != 255
        info_loss = self.ie_cls_head.loss(cls_pred[valid_mask], tmp_labels[valid_mask], prefix='infor_')
        losses.update(info_loss)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    chargrid_map,
                    gt_bboxes=None,
                    rescale=False,
                    **kwargs):
        input_img = chargrid_map if self.use_chargrid else img
        if self.use_chargrid:
            input_img = torch.cat(input_img, dim=0)
        result = {}

        # text detection branch
        img_feat = self.extract_feat(input_img)
        recog_rois = bbox2roi(gt_bboxes[0])
        infor_feats = self.ie_roi_extractor(img_feat[:self.ie_roi_extractor.num_inputs], recog_rois)
        feat, _ = self.process_infor_feats(infor_feats, gt_bboxes[0])
        cls_pred = self.ie_cls_head(feat)
        result['bboxes_labels_pred'] = cls_pred[0].cpu().numpy()

        return [result]