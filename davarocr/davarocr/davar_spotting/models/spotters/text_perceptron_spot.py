"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    text_perceptron_spot.py
# Abstract       :    The main pipeline definition of tp_spot model

# Current Version:    1.0.0
# Date           :    2021-09-01
####################################################################################################
"""
import numpy as np

from .base import SegBasedEndToEnd
from ..builder import SPOTTER


@SPOTTER.register_module()
class TextPerceptronSpot(SegBasedEndToEnd):
    """ Implementation of Text Perceptron spotter model.[1]

    Ref: [1] Text Perceptron: Towards End-to-End Arbitrary Shaped Text Spotting. AAAI-20.
                <https://arxiv.org/abs/2002.06820>`_
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 mask_head=None,
                 rcg_roi_extractor=None,
                 rcg_transformation=None,
                 rcg_backbone=None,
                 rcg_neck=None,
                 rcg_sequence_module=None,
                 rcg_sequence_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """ Network Initialization.

        Args:
            backbone (dict): network backbone (e.g. ResNet)
            neck (dict): network neck (e.g., FPN)
            mask_head (dict): head for loss calculation (e.g., TPHead)
            rcg_roi_extractor (dict): head for extract region of interest (e.g. SingleRoIExtractor)
            rcg_transformation (dict): recognation feature transformation module (e.g. TPS, STN)
            rcg_backbone (dict): backbone of the recognation model (e.g. ResNet)
            rcg_neck (dict): neck of the recognation model (e.g. FPN)
            rcg_sequence_module (dict): module for extract sequence relation (e.g. RNN / BiLSTM/ Transformer)
            rcg_sequence_head (dict): recognition head (e.g., AttentionHead)
            train_cfg (dict): related parameters for training
            test_cfg (dict): related parameters for test
            pretrained (dict): pretrained model
        """
        super().__init__(backbone=backbone,
                         neck=neck,
                         mask_head=mask_head,
                         rcg_roi_extractor=rcg_roi_extractor,
                         rcg_transformation=rcg_transformation,
                         rcg_backbone=rcg_backbone,
                         rcg_neck=rcg_neck,
                         rcg_sequence_module=rcg_sequence_module,
                         rcg_sequence_head=rcg_sequence_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         pretrained=pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_poly_bboxes,
                      gt_masks,
                      gt_texts,
                      **kwargs):
        """ Forward training process and loss computing

        Args:
            img (Tensor): input images
            img_metas (dict) : image meta-info
            gt_poly_bboxes (list(list(np.array))): poly boxes for text instances
            gt_masks (Tensor) : masks for text segmentation
            **kwargs: other parameters

        Returns:
            dict: losses for training data
        """
        losses = dict()

        # ============================= text detection branch ===============================
        feat = self.extract_feat(img)

        # compute features through mask_head
        mask_pred = self.mask_head(feat)

        # get ground-truth label
        mask_targets = self.mask_head.get_target(gt_masks)

        # compute loss
        loss_mask = self.mask_head.loss(mask_pred, mask_targets)

        # update loss
        losses.update(loss_mask)

        # ============================== text recognition branch ==============================
        if sum([len(box) for box in gt_poly_bboxes]) == 0:
            # non positive gts in current batch, we just not train the recognition branch
            return losses

        # Compute fiducial points
        fiducial_points = self.recog_roi_extractor.get_fiducial_points(img, gt_poly_bboxes)

        # Extract feature according to fiducial point
        recog_feats = self.recog_roi_extractor(feat[:self.recog_roi_extractor.num_inputs], fiducial_points)

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

        texts = []
        for text in gt_texts:
            texts += text

        # Text recognition
        recog_target = self.recog_sequence_head.get_target(texts)
        recog_prediction = self.recog_sequence_head(recog_contextual_feature.contiguous(), recog_target, is_train=True)

        loss_recog = self.recog_sequence_head.loss(recog_prediction, recog_target)

        # update loss
        losses.update(loss_recog)

        return losses

    def simple_test(self,
                    img,
                    img_meta,
                    gt_texts=None,
                    **kwargs):
        """Forward inference

        Args:
            img (Tensor): input image
            img_meta (dict): image meta-info
            gt_texts (list(list(str))) : transcriptions for text recognition
            **kwargs: other parameters

        Returns:
            dict: predicted results.  e.g. [{'points':[[x1, y1, ..., xn, yn], ...], 'texts':['apple', ...]}, ...].
        """
        # ============================= text detection branch ===============================
        feat = self.extract_feat(img)
        results = self.mask_head(feat)

        # Post-processing for generating poly bboxes and fiducial points according to masks
        if self.post_processor is not None:
            results = self.post_processor.post_processing(results, img_meta)
            fiducial_points = [result['points'] for result in results]

        # ============================== text recognition branch ==============================
        if sum([len(points) for points in fiducial_points]) == 0:
            # non positive gts in current batch
            results = [{"points" : [], "texts" : []}] * img.size(0)
            return results

        # Compute normalized fiducial points
        fiducial_points = self.recog_roi_extractor.rescale_fiducial_points(img, img_meta, fiducial_points)

        # Extract feature according to fiducial point
        recog_feats = self.recog_roi_extractor(feat[:self.recog_roi_extractor.num_inputs], fiducial_points)

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

        if self.keepdim_test:
            # N x C x 1 x HW -> N x 1 x HW x C
            if self.use_permute_test:
                recog_feats = recog_feats.permute(0, 2, 3, 1)
        else:
            # N x C x 1 x HW -> N x C x HW -> N x HW x C
            recog_feats = recog_feats.squeeze(2).permute(0, 2, 1)

        # Semantic feature extraction
        if self.with_recog_sequence_module:
            recog_contextual_feature = self.recog_sequence_module(recog_feats)
        else:
            recog_contextual_feature = recog_feats

        # Text recognition
        recog_target = self.recog_sequence_head.get_target(gt_texts)
        recog_prediction = self.recog_sequence_head(recog_contextual_feature.contiguous(), recog_target, is_train=False)
        text_preds = self.recog_sequence_head.get_pred_text(recog_prediction, self.test_cfg.batch_max_length)

        # Assign the text recognition result to the corresponding img
        assigner = np.array([i for i in range(len(fiducial_points)) for _ in range(len(fiducial_points[i]))])
        text_preds = np.array(text_preds)
        for batch_id in range(img.size(0)):
            ind = np.where(assigner == batch_id)[0]
            text = text_preds[ind].tolist()
            results[batch_id]['texts'] = text
        return results

