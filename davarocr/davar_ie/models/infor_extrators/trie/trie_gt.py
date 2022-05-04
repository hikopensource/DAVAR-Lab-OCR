"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    trie_with_gt.py
# Abstract       :    TRIEGT implementation with groundtruth of detection and recognition.

# Current Version:    1.0.0
# Date           :    2021-03-19
##################################################################################################
"""
import numpy as np

import torch
from mmdet.core import bbox2roi
from mmdet.models import build_roi_extractor

from davarocr.davar_spotting.models import SPOTTER
from davarocr.davar_common.models.builder import build_embedding

from .trie import TRIE


@SPOTTER.register_module()
class TRIEGT(TRIE):
    """Implementation of TRIE algorithm with groundtruth of detection and recognition [1]

    Ref: [1] TRIE: End-to-End Text Reading and Information Extraction for Document Understanding. ACM MM-20.
                <https://arxiv.org/pdf/2005.13118.pdf>`_
    """
    def __init__(self,
                 backbone,
                 rcg_backbone=None,
                 rcg_roi_extractor=None,
                 rcg_sequence_head=None,
                 infor_context_module=None,
                 infor_node_cls_head=None,
                 infor_sequence_module=None,
                 infor_ner_head=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 rcg_neck=None,
                 rcg_transformation=None,
                 rcg_sequence_module=None,
                 infor_roi_extractor=None,
                 raw_text_embedding=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        """
        Args:
            backbone(dict): backbone of detection model (e.g. ResNet)
            rcg_backbone(dict): backbone of rcg model (e.g. ResNet32)
            rcg_roi_extractor(dict): roi_extractor for rcg
            rcg_sequence_head(dict): sequence head in rcg (e.g. AttentionHead)
            infor_context_module(dict): context module for IE (e.g. MultiModalContextModule)
            infor_node_cls_head(dict): node cls head for IE (e.g. ClsHead)
            infor_sequence_module(dict): config of infor_sequence_module (e.g. LSTM)
            infor_ner_head(dict): ner head for IE
            neck(dict): necks of the detection model (e.g., FPN)
            rpn_head(dict): rpn head
            roi_head(dict): roi head
            rcg_neck(dict): necks of rcg model (default None)
            rcg_transformation(dict): transformation of rcg model
            rcg_sequence_module(dict): sequence module of rcg model (e.g. CascadeRNN)
            raw_text_embedding (dict): original text_embedding config (e.g. Embedding)
            train_cfg(dict): default None
            test_cfg(dict): definition of postprocess/ prune_model
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        super().__init__(
            backbone=backbone,
            rcg_backbone=rcg_backbone,
            rcg_roi_extractor=rcg_roi_extractor,
            rcg_sequence_head=rcg_sequence_head,
            infor_context_module=infor_context_module,
            infor_node_cls_head=infor_node_cls_head,
            infor_sequence_module=infor_sequence_module,
            infor_ner_head=infor_ner_head,
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
        if infor_roi_extractor is not None:
            self.infor_roi_extractor = build_roi_extractor(infor_roi_extractor)
        if raw_text_embedding is not None:
            self.raw_text_embedding = build_embedding(raw_text_embedding)

        self.batch_max_length_train = getattr(self.train_cfg, 'batch_max_length', 30)
        self.batch_max_length_test = getattr(self.test_cfg, 'batch_max_length', 30)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        super().init_weights(pretrained)

        if self.with_raw_text_embedding:
            self.raw_text_embedding.init_weights(pretrained)

    @property
    def with_infor_roi_extractor(self):
        """

        Returns:
            Determine the model with the infor_roi_extractor or not
        """
        return hasattr(self, 'infor_roi_extractor') and self.infor_roi_extractor is not None

    @property
    def with_raw_text_embedding(self):
        """

        Returns:
            Determine the model with the raw_text_embedding or not
        """
        return hasattr(self, 'raw_text_embedding') and self.raw_text_embedding is not None

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_texts,
                      array_gt_texts=None,
                      gt_bieo_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """ Forward train process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_bboxes (list(Tensor): Tensor bboxes for each image, in [x_tl, y_tl, x_br, y_br] order.
            gt_labels (list(Tensor): category labels for each bbox
            gt_texts (list(list(string))): text transcription: e.g. ["abc", "bte",....]
            array_gt_texts (list(np.array)): text transcription after tokenization in np.array format.
            gt_bieo_labels (list(list(list(float))): character-level labels
            gt_bboxes_ignore (list(Tensor)): ignored bboxes
        Returns:
            dict: all losses in a dict
        """

        losses = dict()

        info_labels = gt_labels

        # text detection branch
        img_feat = self.extract_feat(img)
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                img_feat,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # ROI forward and loss
        roi_losses = self.roi_head.forward_train(img_feat, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # text recognition branch
        if sum([box.shape[0] for box in gt_bboxes]) == 0:
            # non positive gts in current batch, we just not train the recognition branch
            return losses

        recog_rois = bbox2roi(gt_bboxes)

        info_feat_list = []
        if self.with_infor_roi_extractor:
            infor_feats = self.infor_roi_extractor(img_feat[:self.infor_roi_extractor.num_inputs], recog_rois, gt_masks)
            info_feat_list.append(infor_feats)

        # text embedding
        if self.with_raw_text_embedding:
            gt_texts_tensor = torch.from_numpy(np.concatenate(array_gt_texts, axis=0)).long().to(img_feat[0].device)

            recog_hidden = self.raw_text_embedding(gt_texts_tensor)
            info_feat_list.append(recog_hidden)

        # multimodal context module
        multimodal_context, batched_img_label, _ = self.infor_context_module(info_feat_list,
                                                                                           pos_feat=gt_bboxes,
                                                                                           img_metas=img_metas,
                                                                                           info_labels=info_labels,
                                                                                           bieo_labels=gt_bieo_labels)

        # node classification task if required
        if self.with_infor_node_cls_head:
            cls_pred = self.infor_node_cls_head(multimodal_context)
            cls_pred = cls_pred.view(-1, cls_pred.size(-1))
            tmp_labels = torch.cat(batched_img_label, 0).view(-1)
            valid_mask = tmp_labels != 255
            info_loss = self.infor_node_cls_head.loss(cls_pred[valid_mask], tmp_labels[valid_mask], prefix='infor_')
            losses.update(info_loss)

        # lstm + crf if required
        if self.with_infor_sequence_module and self.with_infor_ner_head:
            valid_multimodal_context = []
            for idx, per_gt_bbox in enumerate(gt_bboxes):
                valid_length = per_gt_bbox.size(0)
                valid_multimodal_context.append(multimodal_context[idx, :valid_length, :])

            valid_multimodal_context = torch.cat(valid_multimodal_context, 0)[:, None, :].repeat(1,
                                                                                                 recog_hidden.size(1),
                                                                                                 1)

            fused_features = torch.cat((recog_hidden, valid_multimodal_context), -1)

            fused_features = self.infor_sequence_module(fused_features)
            outputs = self.infor_ner_head(fused_features)

            valid_target = [torch.tensor(per, dtype=torch.long).to(fused_features.device) for per in gt_bieo_labels]
            target = torch.cat(valid_target, 0)
            mask = target.lt(255)
            target[target.eq(255)] = 0
            crf_loss = self.infor_ner_head.loss(outputs, target, mask)
            losses.update(crf_loss)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_texts=None,
                    array_gt_texts=None,
                    rescale=False,
                    gt_bieo_labels=None,
                    **kwargs):
        """ Forward test process

        Args:
            img(Tensor): input images
            img_metas(dict): image meta infos
            gt_texts(list(string): text transcription, default to None
            rescale(boolean): if the image be re-scaled
            array_gt_texts (list(np.array)): text transcription after tokenization in np.array format.
            gt_bieo_labels (list(list(list(float))): character-level labels

        Returns:
            list(dict) : predicted result, e.g. ['bboxes': np.array, 'texts': ['abc', 'def' ...],
                    'bboxes_labels_pred': np.array, 'bboxes_bieo_labels_pred': np.array]
        """

        result = {}
        result['bboxes'] = gt_bboxes[0][0].cpu().numpy()
        result['texts'] = gt_texts[0][0]

        # text detection branch
        img_feat = self.extract_feat(img)

        recog_rois = bbox2roi(gt_bboxes[0])

        info_feat_list = []
        if self.with_infor_roi_extractor:
            infor_feats = self.infor_roi_extractor(img_feat[:self.infor_roi_extractor.num_inputs], recog_rois)
            info_feat_list.append(infor_feats)

        # text embedding
        if self.with_raw_text_embedding:
            gt_texts_tensor = torch.from_numpy(np.concatenate(array_gt_texts[0], axis=0)).long().to(img_feat[0].device)

            recog_hidden = self.raw_text_embedding(gt_texts_tensor)
            info_feat_list.append(recog_hidden)

        # multimodal context module
        multimodal_context, _, _ = self.infor_context_module(info_feat_list,
                                                                                                  pos_feat=gt_bboxes[0],
                                                                                                  img_metas=img_metas,
                                                                                                  info_labels=None,
                                                                                                  bieo_labels=None)

        # node classification task if required
        if self.with_infor_node_cls_head:
            cls_pred = self.infor_node_cls_head(multimodal_context)
            result['bboxes_labels_pred'] = cls_pred[0].cpu().numpy()

        # lstm + crf if required
        if self.with_infor_sequence_module and self.with_infor_ner_head:
            valid_multimodal_context = []
            for idx, per_gt_bbox in enumerate(gt_bboxes):
                valid_length = per_gt_bbox[0].size(0)
                valid_multimodal_context.append(multimodal_context[idx, :valid_length, :])
            valid_multimodal_context = torch.cat(valid_multimodal_context, 0)[:, None, :].repeat(1,
                                                                                                 recog_hidden.size(
                                                                                                     1), 1)

            fused_features = torch.cat((recog_hidden, valid_multimodal_context), -1)

            fused_features = self.infor_sequence_module(fused_features)
            outputs = self.infor_ner_head(fused_features)

            valid_target = [torch.tensor(per, dtype=torch.long).to(fused_features.device) for per in gt_bieo_labels[0]]
            target = torch.cat(valid_target, 0)
            mask = target.lt(255)
            tags = self.infor_ner_head.get_predict(outputs, mask=mask)

            result['bboxes_bieo_labels_pred'] = tags[0].cpu().numpy()

        return [result]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Forward aug_test. Not implemented.
        """
        raise NotImplementedError
