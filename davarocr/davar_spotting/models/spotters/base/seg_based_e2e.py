"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    seg_based_e2e.py
# Abstract       :    The main structure definition of segmentation based text spotter

# Current Version:    1.0.0
# Date           :    2021-09-01
####################################################################################################
"""
import torch.nn as nn

from mmdet.models import build_backbone, build_roi_extractor, build_head, build_neck
from davarocr.davar_common.models import build_connect, build_transformation
from davarocr.davar_common.core.builder import build_postprocess
from davarocr.davar_spotting.models.builder import SPOTTER

from .base import BaseEndToEnd


@SPOTTER.register_module()
class SegBasedEndToEnd(BaseEndToEnd):
    """Segmentation-based recognition framework."""

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
            train_cfg (mmcv.Config): model training cfg parameter
            test_cfg (mmcv.Config): model test cfg parameter
            pretrained (str, optional): model path of the pre_trained model
        """
        super().__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if mask_head is not None:
            self.mask_head = build_head(mask_head)

        if rcg_roi_extractor is not None:
            self.recog_roi_extractor = build_roi_extractor(rcg_roi_extractor)

        if rcg_backbone is not None:
            self.recog_backbone = build_backbone(rcg_backbone)
            self.rcg_from_img = rcg_backbone.get('input_channel', 256) in [1, 3]

        if rcg_sequence_head is not None:
            self.recog_sequence_head = build_head(rcg_sequence_head)

        if rcg_transformation is not None:
            self.recog_transformation = build_transformation(rcg_transformation)

        if rcg_neck is not None:
            self.recog_neck = build_neck(rcg_neck)

        if rcg_sequence_module is not None:
            self.recog_sequence_module = build_connect(rcg_sequence_module)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        if hasattr(self.test_cfg, 'postprocess'):
            self.post_processor = build_postprocess(self.test_cfg.postprocess)
        else:
            self.post_processor = None

        # default keep dim is 4
        self.keepdim_train = getattr(self.train_cfg, 'keep_dim', True)
        self.keepdim_test = getattr(self.test_cfg, 'keep_dim', True)

        # default keep dim is 4
        self.use_permute_train = getattr(self.train_cfg, 'use_permute', True)
        self.use_permute_test = getattr(self.test_cfg, 'use_permute', True)

    @property
    def with_recog_roi_extractor(self):
        """
        Returns:
            bool: Determine the model is with the recog_roi_extractor or not
        """

        return hasattr(self, 'recog_roi_extractor') and self.recog_roi_extractor is not None

    @property
    def with_recog_sequence_head(self):
        """
        Returns:
            bool: Determine the model is with the recog_sequence_head or not
        """

        return hasattr(self, 'recog_sequence_head') and self.recog_sequence_head is not None

    @property
    def with_recog_sequence_module(self):
        """
        Returns:
            bool: Determine the model is with the recog_sequence_module or not
        """

        return hasattr(self, 'recog_sequence_module') and self.recog_sequence_module is not None

    @property
    def with_recog_transformation(self):
        """
        Returns:
            bool: Determine the model is with the recog_transformation or not
        """

        return hasattr(self, 'recog_transformation') and self.recog_transformation is not None

    @property
    def with_recog_neck(self):
        """
        Returns:
            bool: Determine the model is with the recog_neck or not
        """

        return hasattr(self, 'recog_neck') and self.recog_neck is not None

    @property
    def with_recog_backbone(self):
        """
        Returns:
            bool: Determine the model is with the recog_backbone or not
        """

        return hasattr(self, 'recog_backbone') and self.recog_backbone is not None

    def init_weights(self, pretrained=None):
        """Parameters initialization

        Args:
            pretrained(dict): pretrained model
        """
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)

        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for module in self.neck:
                    module.init_weights()
            else:
                self.neck.init_weights()

        self.mask_head.init_weights()

        if self.with_recog_roi_extractor:
            self.recog_roi_extractor.init_weights()

        if self.with_recog_backbone:
            self.recog_backbone.init_weights()

        if self.with_recog_sequence_head:
            self.recog_sequence_head.init_weights()

        if self.with_recog_transformation:
            self.recog_transformation.init_weights()

        if self.with_recog_neck:
            self.recog_neck.init_weights()

        if self.with_recog_sequence_module:
            self.recog_sequence_module.init_weights()

    def extract_feat(self, img):
        """ Feature extraction, mainly including backbone part and neck part

        Args:
            img(Tensor): input image

        Returns:
            Tensor: output feature maps through feature extractor
        """
        feat = self.backbone(img)
        if self.with_neck:
            feat = self.neck(feat)
        return feat

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
        # Extract feature
        recog_feats = self.recog_roi_extractor(feat[:self.recog_roi_extractor.num_inputs], mask_targets)

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

    def forward_dummy(self, img):
        """Dummy forward pass (mainly for FLOPS calculation)

        Args:
            img (Tensor): input image.

        Returns:
            obj: feature map output
        """
        feat = self.extract_feat(img)
        outs = self.mask_head(feat)
        return outs

    def simple_test(self,
                    img,
                    img_meta,
                    gt_texts=None,
                    **kwargs):
        """Forward inference

        Args:
            img(Tensor): input image
            img_meta(dict): image meta-info
            gt_texts (list(list(str))) : transcriptions for text recognition
            **kwargs: other parameters

        Returns:
            dict: predicted results.  e.g. [{'points':[[x1, y1, ..., xn, yn], ...], 'texts':['apple', ...]}, ...].
        """
        # ============================= text detection branch ===============================
        feat = self.extract_feat(img)
        mask_preds = self.mask_head(feat)

        # ============================== text recognition branch ==============================
        # Extract feature
        recog_feats = self.recog_roi_extractor(feat[:self.recog_roi_extractor.num_inputs], mask_preds)

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

        results = dict()
        results['mask_preds'] = mask_preds
        results['text_preds'] = text_preds

        if self.post_processor is not None:
            results = self.post_processor.post_processing(results, img_meta)

        return results

    def aug_test(self, img, img_meta):
        """ Forward aug_test. Not implemented."""
        raise NotImplementedError
