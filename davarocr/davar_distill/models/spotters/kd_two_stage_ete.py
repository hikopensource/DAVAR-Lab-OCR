"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    kd_two_stage_ete.py
# Abstract       :    The main pipeline definition of two stage end-to-end spotter for distillation

# Current Version:    1.0.0
# Date           :    2022-07-07
##################################################################################################
"""
import torch

from mmcv.runner import auto_fp16
from mmdet.core import bbox2roi
from davarocr.davar_spotting.models import SPOTTER
from davarocr.davar_spotting.models import MaskRCNNSpot


@SPOTTER.register_module()
class KDTwoStageEndToEnd(MaskRCNNSpot):
    """ Two stage recognition framework for distillation """

    def __init__(self,
                 backbone,
                 rcg_roi_extractor,
                 rcg_sequence_head,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 rcg_backbone=None,
                 rcg_neck=None,
                 rcg_transformation=None,
                 rcg_sequence_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """
        Args:
            backbone (dict): backbone of the model (e.g. ResNet)
            rcg_roi_extractor (dict): head for extract region of interest (e.g. SingleRoIExtractor)
            rcg_sequence_head (dict): recognition head (e.g., AttentionHead)
            neck (dict): neck of the model (e.g., FPN)
            rpn_head (dict): head for generate proposal (e.g. RPNHead)
            roi_head (dict): head for predict mask/box according to roi (e.g. StandardRoIHead)
            rcg_backbone (dict): backbone of the recognation model (e.g. ResNet)
            rcg_neck (dict): neck of the recognation model (e.g. FPN)
            rcg_transformation (dict): recognation feature transformation module (e.g. TPS, STN)
            rcg_sequence_module (dict): module for extract sequence relation
            train_cfg (mmcv.Config): model training cfg parameter
            test_cfg (mmcv.Config): model test cfg parameter
            pretrained (str, optional): model path of the pre_trained model
        """
        super().__init__(
            backbone=backbone,
            rcg_backbone=rcg_backbone,
            rcg_roi_extractor=rcg_roi_extractor,
            rcg_sequence_head=rcg_sequence_head,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            rcg_neck=rcg_neck,
            rcg_transformation=rcg_transformation,
            rcg_sequence_module=rcg_sequence_module,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )


    def forward_train(self,
                      img,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_texts=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      is_train=True,
                      **kwargs):
        """ Forward train process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_bboxes (list(list(float))): bounding boxes for text instances:
                                           e.g. [[x1_1, y1_1, ....],....,[x_n_1, y_n_1,...]]
            gt_labels (list(list(str))): labels for classification/detection:
                                         e.g. [['title'], ['code'], ['num'], ...]
            gt_texts (list(str)): transcriptions for text recognition:
                                  e.g. ["apple", "mango",....]
            gt_bboxes_ignore (list(list(float))): ignored bounding boxes:
                                                  e.g. [[x1_1, y1_1, ....],....,[x_n_1, y_n_1,...]]
            gt_masks (list(BitmapMasks)): masks for text segmentation:
                                          e.g. [BitmapMasks(num_masks=num, height=h, width=w), ...]
            proposals (list(list(float))): proposals for detection:
                                           e.g. [[x1_1, y1_1, ....],....,[x_n_1, y_n_1,...]]
            **kwargs: other parameters

        Returns:
            dict: all losses in a dict
        """

        losses = dict()

        # ===================== text detection branch ====================
        # Feature extraction
        feats = self.extract_feat(img)

        if is_train:
            # RPN forward
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    feats,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
                losses.update(rpn_losses)
            else:
                proposal_list = proposals

            # RoI forward
            roi_losses = self.roi_head.forward_train(feats, img_metas, proposal_list,
                                                     gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks)

            losses.update(roi_losses)

        # ===================== text recognition branch ====================
        if sum([box.shape[0] for box in gt_bboxes]) == 0:
            # non positive gts in current batch, we just not train the recognition branch
            return losses

        recog_rois = bbox2roi(gt_bboxes)

        # Extract feature according to RoI
        recog_feats = self.recog_roi_extractor(
            feats[:self.recog_roi_extractor.num_inputs], recog_rois, gt_masks)
        losses.update({"rcg_roi_feat" : recog_feats})

        # Feature transformation
        if self.with_recog_transformation:
            recog_feats = self.recog_transformation(recog_feats)

        # Further extract feature for text recognation
        if self.with_recog_backbone:
            recog_feats = self.recog_backbone(recog_feats)

        # Extract multi-scale feature
        if self.with_recog_neck:
            recog_feats = self.recog_neck(recog_feats)

        batch, old_c, old_h, old_w = recog_feats.size()
        recog_feats = recog_feats.view(batch, old_c, 1, old_h * old_w)

        if self.keepdim_train:
            # N x C x 1 x HW -> N x 1 x HW x C
            if self.use_permute_train:
                recog_feats = recog_feats.permute(0, 2, 3, 1)
        else:
            # N x C x 1 x HW -> N x C x HW -> N x HW x C
            recog_feats = recog_feats.squeeze(2).permute(0, 2, 1)

        # Semantic feature extraction
        if self.with_recog_sequence_module:
            recog_contextual_feature = self.recog_sequence_module(recog_feats)
        else:
            recog_contextual_feature = recog_feats
        losses.update({"rcg_context_feat" : recog_contextual_feature})

        texts = []
        for text in gt_texts:
            texts += text

        # Text recognition
        recog_target = self.recog_sequence_head.get_target(texts)
        recog_prediction = self.recog_sequence_head(recog_contextual_feature.contiguous(), recog_target)

        if is_train:
            loss_recog = self.recog_sequence_head.loss(recog_prediction, recog_target)
            losses.update(loss_recog)
        losses.update({"rcg_pred" : recog_prediction})
        losses.update({"recog_target" : recog_target})
        return losses

    def forward_dummy(self,
                      img,
                      **kwargs):
        """ Forward dummy for calculate flops.

        Args:
            img (Tensor): input images
            **kwargs: other parameters

        Returns:
            dict: dummy output
        """
        outs = ()
        feats = self.extract_feat(img)
        if self.with_rpn:
            rpn_outs = self.rpn_head(feats)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        self.roi_head.forward_dummy(feats, proposals)
        recog_feats = self.recog_roi_extractor.forward_dummy(feats[:self.recog_roi_extractor.num_inputs])
        recog_feats = self.recog_backbone(recog_feats)
        batch, old_c, old_h, old_w = recog_feats.size()
        recog_feats = recog_feats.view(batch, old_c, 1, old_h * old_w)
        if self.keepdim_test:
            # N x C x 1 x HW -> N x 1 x HW x C
            if self.use_permute_test:
                recog_feats = recog_feats.permute(0, 2, 3, 1)
        else:
            # N x C x 1 x HW -> N x C x HW -> N x HW x C
            recog_feats = recog_feats.squeeze(2).permute(0, 2, 1)
        recog_contextual_feature = self.recog_sequence_module(recog_feats)
        self.recog_sequence_head(recog_contextual_feature.contiguous(), None, is_train=False)
        return outs
