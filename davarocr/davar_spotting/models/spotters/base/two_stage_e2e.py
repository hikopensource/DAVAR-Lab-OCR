"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    two_stage_e2e.py
# Abstract       :    The main pipeline definition of two stage end-to-end spotter

# Current Version:    1.0.0
# Date           :    2021-03-19
##################################################################################################
"""
import torch
import torch.nn as nn
import numpy as np

from mmdet.models import build_backbone, build_roi_extractor, build_head, build_neck
from mmdet.core import bbox2roi

from davarocr.davar_common.models import build_connect, build_transformation
from davarocr.davar_common.core.builder import build_postprocess
from davarocr.davar_spotting.models.builder import SPOTTER

from .base import BaseEndToEnd


@SPOTTER.register_module()
class TwoStageEndToEnd(BaseEndToEnd):
    """ Two stage recognition framework. """

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
                 pretrained=None,):
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
            rcg_sequence_module (dict): module for extract sequence relation (e.g. RNN / BiLSTM/ Transformer)
            train_cfg (mmcv.Config): model training cfg parameter
            test_cfg (mmcv.Config): model test cfg parameter
            pretrained (str, optional): model path of the pre_trained model
        """

        super().__init__()

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

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

    @property
    def with_rpn(self):
        """
        Returns:
            bool: Determine the model is with the rpn_head or not
        """

        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """

        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)

        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for module in self.neck:
                    module.init_weights()
            else:
                self.neck.init_weights()

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

    def extract_feat(self, imgs):
        """ Feature extraction, mainly including backbone part and neck part

        Args:
            imgs (Tensor): input images

        Returns:
            list[Tensor]: output feature maps through feature extractor, in different feature map levels,
                          e.g. [4x, 8x, 16x, 32x, 64x]
        """
        feat = self.backbone(imgs)
        if self.with_neck:
            feat = self.neck(feat)
        return feat

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_texts,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """ Forward train process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_bboxes (list(list(np.array))): bounding boxes for text instances:
            gt_labels (list(list(str))): labels for classification/detection:
            gt_texts (list(list(str))): transcriptions for text recognition:
            gt_bboxes_ignore (list(list(np.array))): ignored bounding boxes:
            gt_masks (list(BitmapMasks)): masks for text segmentation:
                                          e.g. [BitmapMasks(num_masks=num, height=h, width=w), ...]
            proposals (list(list(np.array))): proposals for detection:
            **kwargs: other parameters

        Returns:
            dict: all losses in a dict
        """

        losses = dict()

        # ===================== text detection branch ====================
        # Feature extraction
        feats = self.extract_feat(img)

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
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        losses.update(roi_losses)

        # ===================== text recognition branch ====================
        if sum([box.shape[0] for box in gt_bboxes]) == 0:
            # non positive gts in current batch, we just not train the recognition branch
            return losses

        recog_rois = bbox2roi(gt_bboxes)

        # Extract feature according to RoI
        recog_feats = self.recog_roi_extractor(
            feats[:self.recog_roi_extractor.num_inputs], recog_rois, gt_masks)

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

        losses.update(loss_recog)
        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    gt_texts=None,
                    rescale=False,
                    proposals=None):
        """ Forward test process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_texts (list(list(str))): transcriptions for text recognition:
            rescale (boolean): if the image be re-scaled
            proposals (list(list(np.array))): proposals for detection:
            **kwargs: other parameters

        Returns:
            dict: formated inference results
        """

        # ===========================text detection branch================
        results = dict()

        # Feature extraction
        feats = self.extract_feat(img)

        # Get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals

        # Get text detection results (including bboxes, masks coresponding to each text instance)
        det_bboxes = self.roi_head.simple_test(feats, proposal_list, img_metas, rescale=rescale)

        results['bboxes_preds'] = det_bboxes # list(np.array(N, 5)), len=B

        if sum([box.shape[0] for box in det_bboxes]) == 0:
            # non positive gts in current batch
            results = [{"points" : [], "texts" : []}] * img.size(0)
            return results

        # Rescale det_bboxes and det_masks
        if rescale:
            _bboxes = []
            for idx, per_bbox in enumerate(det_bboxes):
                per_bbox = torch.from_numpy(per_bbox[:, :4] * img_metas[idx]['scale_factor']).to(img.device)
                _bboxes.append(per_bbox)
        else:
            _bboxes = [torch.from_numpy(per_bbox[:, :4]).to(img.device) for per_bbox in det_bboxes]

        recog_rois = bbox2roi(_bboxes)

        # Extract feature according to RoI and masks
        recog_feats = self.recog_roi_extractor(
            feats[:self.recog_roi_extractor.num_inputs], recog_rois)

        # Feature transformation
        if self.with_recog_transformation:
            recog_feats = self.recog_transformation(recog_feats)

        # Further extract feature for text recognation
        if self.with_recog_backbone:
            recog_feats = self.recog_backbone(recog_feats)

        # Extract multi-scale feature
        if self.with_recog_neck:
            recog_feats = self.recog_neck(recog_feats)

        # N x C x H x W -> N x C x 1 x HW
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
        det_texts = []
        recog_rois = recog_rois.cpu().numpy()
        text_preds = np.array(text_preds)
        for batch_id in range(img.size(0)):
            ind = np.where(recog_rois == batch_id)[0]
            text = text_preds[ind].tolist()
            det_texts.append(text)

        results['text_preds'] = det_texts  # list(B, N)

        # Post-processing
        if self.post_processor is not None:
            results = self.post_processor.post_processing(results)

        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        """ Forward aug_test. Not implemented."""
        raise NotImplementedError
