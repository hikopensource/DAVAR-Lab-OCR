"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    VSR.py
# Abstract       :    VSR implementation

# Current Version:    1.0.1
# Date           :    2022-04-20
######################################################################################################
"""

import torch
import mmcv
from torch import nn
import numpy as np

from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_roi_extractor
from mmdet.models.detectors.base import BaseDetector
from mmdet.core import bbox2roi
from mmdet.core.visualization import imshow_det_bboxes
from davarocr.davar_common.models import build_connect, build_embedding


@DETECTORS.register_module()
class VSR(BaseDetector):
    """Implementation of VSR algorithm [1]

    Ref: [1] VSR: A Unified Framework for Document Layout Analysis combining Vision, Sementics and Relations.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 chargrid=None,
                 bertgrid=None,
                 sentencegrid=None,
                 backbone_semantic=None,
                 multimodal_merge=None,
                 line_roi_extractor=None,
                 line_gcn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """
        Args:
            backbone(dict): backbone of detection model (e.g. ResNet)
            neck(dict): neck of the detection model (e.g., FPN)
            rpn_head(dict): rpn head
            roi_head(dict): roi head
            chargrid(dict): chargrid configuration
            sentencegrid(dict): sentencegrid configuration
            backbone_semantic(dict): backbone of detection model for semantic branch (e.g. ResNet)
            multimodal_merge(dict): multimodal merge config
            line_roi_extractor(dict): roi extractor for word/ line
            line_gcn_head(dict): classification head on word/ line
            train_cfg(dict): default None
            test_cfg(dict): definition of postprocess/ prune_model
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        super().__init__()

        # merge params
        self.multimodal_merge = multimodal_merge
        input_channels = 0

        # chargrid
        if chargrid is not None:
            self.chargrid_embedding = build_embedding(chargrid)
            input_channels += chargrid.get('embedding_dim')

        # bertgrid
        if bertgrid is not None:
            self.bertgrid_embedding = build_embedding(bertgrid)
            input_channels += bertgrid.get('embedding_dim')

        # sentencegrid
        if sentencegrid is not None:
            self.sentencegrid_embedding = build_embedding(sentencegrid)
            input_channels += sentencegrid.get('embedding_dim')

        # semantic branch only
        if backbone_semantic is None:
            # whether to incorporate img feat
            if self.multimodal_merge.get('with_img_feat'):
                backbone.update(in_channels=input_channels + 3)
            else:
                backbone.update(in_channels=input_channels)
        else:
            # semantic branch + vision branch
            assert backbone.get('in_channels', 3) == 3
            backbone_semantic.update(in_channels=input_channels)
            self.backbone_semantic = build_backbone(backbone_semantic)

            multimodal_feat_merge = self.multimodal_merge.get('multimodal_feat_merge')
            self.multimodal_feat_merge = build_connect(multimodal_feat_merge)

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        # nlp layout analysis task: token classification
        if line_roi_extractor is not None and line_gcn_head is not None:
            self.line_roi_extractor = build_roi_extractor(line_roi_extractor)
            self.line_gcn_head = build_head(line_gcn_head)

        # cv layout analysis task: detection/ segmentation
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_chargrid_embedding(self):
        """

        Returns:
            bool: Determine the model with the chargrid_embedding or not
        """
        return hasattr(self, 'chargrid_embedding') and self.chargrid_embedding is not None

    @property
    def with_bertgrid_embedding(self):
        """

        Returns:
            bool: Determine the model with the bertgrid_embedding or not
        """
        return hasattr(self, 'bertgrid_embedding') and self.bertgrid_embedding is not None

    @property
    def with_sentencegrid_embedding(self):
        """

        Returns:
            bool: Determine the model with the sentencegrid_embedding or not
        """
        return hasattr(self, 'sentencegrid_embedding') and self.sentencegrid_embedding is not None

    @property
    def with_line_gcn_head(self):
        """

        Returns:
            bool: Determine the model with the line_gcn_head or not
        """
        return hasattr(self, 'line_gcn_head') and self.line_gcn_head is not None

    @property
    def with_backbone_semantic(self):
        """

        Returns:
            bool: Determine the model with the backbone_semantic or not
        """
        return hasattr(self, 'backbone_semantic') and self.backbone_semantic is not None

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, imgs):
        """Extract img feats.

        Args:
            imgs(Tensor): input imgs

        Returns:
            list[Tensor]: img feats corresponding to different out_indices.
        """
        feat = self.backbone(imgs)
        return feat

    def extract_feat_semantic(self, img):
        """Extract semantic feats.

        Args:
            img(Tensor): input semantic feature maps.

        Returns:
            list[Tensor]: semantic feats corresponding to different out_indices.
        """
        feat = self.backbone_semantic(img)
        return feat

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_texts=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_bboxes_2=None,
                      gt_labels_2=None,
                      gt_bboxes_ignore_2=None,
                      gt_masks_2=None,
                      gt_ctexts=None,
                      gt_cbboxes=None,
                      **kwargs):
        """ Forward train process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_bboxes (list(Tensor): Tensor bboxes for each image, in [x_tl, y_tl, x_br, y_br] order.
                In token granularity.
            gt_texts (list(list(string))): text transcription: e.g. ["abc", "bte",....]
                In token granularity.
            gt_labels (list(Tensor): category labels for each bbox. In token granularity.
            gt_bboxes_ignore (list(Tensor)): ignored bboxes. In token granularity.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. In token granularity.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
            gt_bboxes_2 (list(Tensor): Tensor bboxes for each image, in [x_tl, y_tl, x_br, y_br] order.
                In layout granularity.
            gt_labels_2 (list(Tensor): category labels for each bbox. In layout granularity.
            gt_bboxes_ignore_2 (list(Tensor)): ignored bboxes. In layout granularity.
            gt_masks_2 (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. In layout granularity.
            gt_ctexts (list(list(int))): category ids for each character in an image.
            gt_cbboxes (list(list(array)): Array bboxes for each character in an image.
        Returns:
            dict: all losses in a dict
        """
        # create grid embedding feature maps of different granularity.
        xxgrid = []

        # chargrid
        if self.with_chargrid_embedding:
            chargrid = self.chargrid_embedding(img, gt_ctexts, gt_cbboxes)
            xxgrid.append(chargrid)

        # bertgrid
        if self.with_bertgrid_embedding:
            bertgrid = self.bertgrid_embedding(img, gt_bboxes, gt_texts)
            xxgrid.append(bertgrid)

        # sentencegrid
        if self.with_sentencegrid_embedding:
            sentencegrid = self.sentencegrid_embedding(img, gt_bboxes, gt_texts)
            xxgrid.append(sentencegrid)

        chargrid_map = torch.cat(xxgrid, 1)

        # semantic branch only
        if not self.with_backbone_semantic:
            if self.multimodal_merge.get('with_img_feat'):
                input_tensors = torch.cat([img, chargrid_map], 1)
                feat = self.extract_feat(input_tensors)
            else:
                feat = self.extract_feat(chargrid_map)
            # FPN
            if self.with_neck:
                feat = self.neck(feat)
        else:
            # semantic branch + visual branch
            visual_feat = self.extract_feat(img)
            semantic_feat = self.extract_feat_semantic(chargrid_map)

            # multimodal fusion
            feat = self.multimodal_feat_merge(visual_feat, semantic_feat)

            # FPN
            if self.with_neck:
                feat = self.neck(feat)

        losses = dict()

        # nlp layout analysis task: token classification
        if self.with_line_gcn_head:
            rois = bbox2roi(gt_bboxes)
            line_feats = self.line_roi_extractor(feat[:self.line_roi_extractor.num_inputs], rois)

            line_cls_score = self.line_gcn_head(line_feats, rois, img_metas)
            line_targets = self.line_gcn_head.get_targets(gt_labels)

            loss_line_cls = self.line_gcn_head.loss(line_cls_score, line_targets)
            losses.update(loss_line_cls)
            return losses

        # cv layout analysis task: detection/ segmentation
        if not self.train_cfg.box_level:
            gt_bboxes = gt_bboxes_2
            gt_labels = gt_labels_2
            gt_bboxes_ignore = gt_bboxes_ignore_2
            gt_masks = gt_masks_2

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                feat,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if self.with_roi_head:
            roi_losses = self.roi_head.forward_train(feat, img_metas, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_texts=None,
                    proposals=None,
                    gt_ctexts=None,
                    gt_cbboxes=None,
                    rescale=False):
        """ Forward test process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_bboxes (list(Tensor): Tensor bboxes for each image, in [x_tl, y_tl, x_br, y_br] order.
                In token granularity.
            gt_texts (list(list(string))): text transcription: e.g. ["abc", "bte",....]
                In token granularity.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
            gt_ctexts (list(list(int))): category ids for each character in an image.
            gt_cbboxes (list(list(array)): Array bboxes for each character in an image.
        Returns:
            list: prediction result for each image.
        """
        xxgrid = []

        # chargrid
        if self.with_chargrid_embedding:
            chargrid = self.chargrid_embedding(img, gt_ctexts[0], gt_cbboxes[0])
            xxgrid.append(chargrid)

        # bertgrid
        if self.with_bertgrid_embedding:
            bertgrid = self.bertgrid_embedding(img, gt_bboxes[0], gt_texts[0])
            xxgrid.append(bertgrid)

        # sentencegrid
        if self.with_sentencegrid_embedding:
            sentencegrid = self.sentencegrid_embedding(img, gt_bboxes[0], gt_texts[0])
            xxgrid.append(sentencegrid)

        chargrid_map = torch.cat(xxgrid, 1)

        # semantic branch only
        if not self.with_backbone_semantic:
            if self.multimodal_merge.get('with_img_feat'):
                input_tensors = torch.cat([img, chargrid_map], 1)
                feat = self.extract_feat(input_tensors)
            else:
                feat = self.extract_feat(chargrid_map)
            # FPN
            if self.with_neck:
                feat = self.neck(feat)
        else:
            # semantic branch + visual branch
            visual_feat = self.extract_feat(img)
            semantic_feat = self.extract_feat_semantic(chargrid_map)

            # multimodal fusion
            feat = self.multimodal_feat_merge(visual_feat, semantic_feat)

            if self.with_neck:
                feat = self.neck(feat)

        # nlp layout analysis task: token classification
        if self.with_line_gcn_head:
            result = []
            rois = bbox2roi(gt_bboxes[0])
            line_feats = self.line_roi_extractor(feat[:self.line_roi_extractor.num_inputs], rois)

            line_cls_score = self.line_gcn_head(line_feats, rois, img_metas)

            # det format
            bboxes = gt_bboxes[0][0]
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
            if bboxes.size(0) > 0:
                scale_factor = bboxes.new_tensor(scale_factors)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                    bboxes.size()[0], -1)
            tmp_result = torch.cat((bboxes, torch.nn.functional.softmax(line_cls_score, -1)), -1)
            result.append(tmp_result)

            # result.append(torch.nn.functional.softmax(line_cls_score, -1).cpu().numpy())
            return tuple(result)

        # cv layout analysis task: detection/ segmentation
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feat, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(feat, proposal_list, img_metas, rescale=rescale)

    def aug_test(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_labels=None,
                    gt_bboxes_ignore=None,
                    gt_masks=None,
                    proposals=None,
                    input_ids=None,
                    gt_bboxes_2=None,
                    gt_labels_2=None,
                    gt_bboxes_ignore_2=None,
                    gt_masks_2=None,
                    token_type_ids=None,
                    attention_mask=None,
                    gt_ctexts=None,
                    gt_cattributes=None,
                    gt_cbboxes=None,
                    in_bboxes_2=None,
                    rescale=False):
        """Forward aug_test. Not implemented.
        """
        raise NotImplementedError

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        if self.with_roi_head:
            super().show_result(img,
                                result,
                                score_thr=score_thr,
                                bbox_color=bbox_color,
                                text_color=text_color,
                                mask_color=mask_color,
                                thickness=thickness,
                                font_size=font_size,
                                win_name=win_name,
                                show=show,
                                wait_time=wait_time,
                                out_file=out_file)
        else:
            img = mmcv.imread(img)
            img = img.copy()

            bbox_result = result.cpu().numpy()
            bboxes = np.concatenate([bbox_result[:, :4], np.ones((bbox_result.shape[0], 1))], -1)
            labels = np.argmax(bbox_result[:, 4:], -1)
            # if out_file specified, do not show image in window
            if out_file is not None:
                show = False
            # draw bounding boxes
            img = imshow_det_bboxes(
                img,
                bboxes,
                labels,
                None,
                class_names=self.CLASSES,
                score_thr=score_thr,
                bbox_color=bbox_color,
                text_color=text_color,
                mask_color=mask_color,
                thickness=thickness,
                font_size=font_size,
                win_name=win_name,
                show=show,
                wait_time=wait_time,
                out_file=out_file)

            if not (show or out_file):
                return img
