"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    trie.py
# Abstract       :    TRIE implementation

# Current Version:    1.0.0
# Date           :    2021-05-20
######################################################################################################
"""
import numpy as np

import torch
from mmdet.core import bbox2roi
from mmdet.models import build_head

from davarocr.davar_spotting.models import SPOTTER
from davarocr.davar_spotting.models import TwoStageEndToEnd
from davarocr.davar_common.models.builder import build_connect


@SPOTTER.register_module()
class TRIE(TwoStageEndToEnd):
    """Implementation of TRIE algorithm [1]

    Ref: [1] TRIE: End-to-End Text Reading and Information Extraction for Document Understanding. ACM MM-20.
                <https://arxiv.org/pdf/2005.13118.pdf>`_
    """
    def __init__(self,
                 backbone,
                 rcg_backbone,
                 rcg_roi_extractor,
                 rcg_sequence_head,
                 infor_context_module,
                 infor_node_cls_head=None,
                 infor_sequence_module=None,
                 infor_ner_head=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 rcg_neck=None,
                 rcg_transformation=None,
                 rcg_sequence_module=None,
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
            train_cfg(dict): default None
            test_cfg(dict): definition of postprocess/ prune_model
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
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
        self.infor_context_module = build_connect(infor_context_module)
        if infor_node_cls_head is not None:
            self.infor_node_cls_head = build_head(infor_node_cls_head)

        if infor_sequence_module is not None:
            self.infor_sequence_module = build_connect(infor_sequence_module)

        if infor_ner_head is not None:
            self.infor_ner_head = build_head(infor_ner_head)

        self.init_weights(pretrained=pretrained)

    @property
    def with_infor_context_module(self):
        """

        Returns:
            Determine the model with the infor_context_module or not
        """
        return hasattr(self, 'infor_context_module') and self.infor_context_module is not None

    @property
    def with_infor_node_cls_head(self):
        """

        Returns:
            Determine the model with the infor_node_cls_head or not
        """
        return hasattr(self, 'infor_node_cls_head') and self.infor_node_cls_head is not None

    @property
    def with_infor_sequence_module(self):
        """

        Returns:
            Determine the model with the infor_sequence_module or not
        """
        return hasattr(self, 'infor_sequence_module') and self.infor_sequence_module is not None

    @property
    def with_infor_ner_head(self):
        """

        Returns:
            Determine the model with the infor_ner_head or not
        """
        return hasattr(self, 'infor_ner_head') and self.infor_ner_head is not None

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        super().init_weights(pretrained)

        if self.with_infor_context_module:
            self.infor_context_module.init_weights(pretrained)
        if self.with_infor_sequence_module:
            self.infor_sequence_module.init_weights(pretrained)
        if self.with_infor_ner_head:
            self.infor_ner_head.init_weights(pretrained)
        if self.with_infor_node_cls_head:
            self.infor_node_cls_head.init_weights(pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_texts,
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
            gt_bieo_labels (list(list(list(float))): character-level labels
            gt_bboxes_ignore (list(Tensor)): ignored bboxes
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
        Returns:
            dict: all losses in a dict
        """
        losses = dict()

        info_labels = gt_labels

        # text detection branch
        x = self.extract_feat(img)
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # ROI forward and loss
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # text recognition branch
        if sum([box.shape[0] for box in gt_bboxes]) == 0:
            # non positive gts in current batch, we just not train the recognition branch
            return losses

        recog_rois = bbox2roi(gt_bboxes)

        # roialign-ed features for rcg and ie
        info_feat_list = []
        recog_feats = self.recog_roi_extractor(
            x[:self.recog_roi_extractor.num_inputs], recog_rois)
        info_feat_list.append(recog_feats)

        # rcg transformation
        if self.with_recog_transformation:
            recog_feats = self.recog_transformation(recog_feats)

        # rcg backbone
        recog_x = self.recog_backbone(recog_feats)

        # rcg necks
        if self.with_recog_neck:
            recog_x = self.recog_neck(recog_x)

        N, old_C, old_H, old_W = recog_x.size()
        recog_x = recog_x.view(N, old_C, 1, old_H * old_W)

        # B,C,1,W -> B,C,W -> L,B,C ? B,L,C/B,1,L,C
        if self.keepdim_train:
            if self.use_permute_train:
                recog_x = recog_x.permute(0, 2, 3, 1)
        else:
            recog_x = recog_x.squeeze(2).permute(0, 2, 1)

        # rcg sequence module
        if self.with_recog_sequence_module:
            recog_contextual_feature = self.recog_sequence_module(recog_x)
        else:
            recog_contextual_feature = recog_x

        texts = []
        for text in gt_texts:
            texts += text

        # rcg sequence head
        recog_target = self.recog_sequence_head.get_target(texts)
        recog_prediction, recog_hidden = self.recog_sequence_head(recog_contextual_feature.contiguous(), recog_target,
                                                                  is_train=True, return_hidden=True)
        loss_recog = self.recog_sequence_head.loss(recog_prediction, recog_target)

        losses.update(loss_recog)

        # remove last hidden EOS to be compatible with bieo labels.
        recog_hidden = recog_hidden[:, :-1, :]

        # information extraction branch
        info_feat_list.append(recog_hidden)

        # multimodal context module
        multimodal_context, batched_img_label, _ = self.infor_context_module(info_feat_list,
                                                                                           pos_feat=gt_bboxes,
                                                                                           img_metas=img_metas,
                                                                                           info_labels=info_labels,
                                                                                           bieo_labels=gt_bieo_labels)

        # node classification head if required
        if self.with_infor_node_cls_head:
            cls_pred = self.infor_node_cls_head(multimodal_context)
            cls_pred = cls_pred.view(-1, cls_pred.size(-1))
            tmp_labels = torch.cat(batched_img_label, 0).view(-1)
            valid_mask = tmp_labels != 255
            info_loss = self.infor_node_cls_head.loss(cls_pred[valid_mask], tmp_labels[valid_mask],
                                                            prefix='infor_')
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
                    gt_texts=None,
                    rescale=False,
                    proposals=None,
                    **kwargs):
        """ Forward test process

        Args:
            img(Tensor): input images
            img_metas(dict): image meta infos
            gt_texts(list(string): text transcription, default to None
            rescale(boolean): if the image be re-scaled
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            list(dict) : predicted result, e.g. ['bboxes': np.array, 'texts': ['abc', 'def' ...],
                    'bboxes_labels_pred': np.array, 'bboxes_bieo_labels_pred': np.array]
        """
        result = {}

        # text detection branch
        x = self.extract_feat(img)
        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

        det_bboxes = [np.concatenate(box, axis=0) for box in results]
        if sum([box.shape[0] for box in det_bboxes]) == 0:
            # non positive gts in current batch, we just not train the recognition branch
            return results + (None, )

        if rescale:
            _bboxes = [torch.from_numpy(per_bbox[:, :4] * img_metas[idx]['scale_factor']).to(img.device) for
                       idx, per_bbox in enumerate(det_bboxes)]
        else:
            _bboxes = [torch.from_numpy(per_bbox[:, :4]).to(img.device) for per_bbox in det_bboxes]

        result['bboxes'] = _bboxes[0].cpu().numpy()

        recog_rois = bbox2roi(_bboxes)

        info_feat_list = []
        recog_feats = self.recog_roi_extractor(
                x[:self.recog_roi_extractor.num_inputs], recog_rois)
        info_feat_list.append(recog_feats)

        if self.with_recog_transformation:
            recog_feats = self.recog_transformation(recog_feats)

        recog_x = self.recog_backbone(recog_feats)

        if self.with_recog_neck:
            recog_x = self.recog_neck(recog_x)

        N, old_C, old_H, old_W = recog_x.size()
        recog_x = recog_x.view(N, old_C, 1, old_H * old_W)
        if self.keepdim_test:
            if self.use_permute_test:
                recog_x = recog_x.permute(0, 2, 3, 1)
        else:
            recog_x = recog_x.squeeze(2).permute(0, 2, 1)
        # B,C,1,W -> B,C,W -> L,B,C ? B,L,C/B,1,L,C

        if self.with_recog_sequence_module:
            recog_contextual_feature = self.recog_sequence_module(recog_x)
        else:
            recog_contextual_feature = recog_x

        recog_target = self.recog_sequence_head.get_target(gt_texts)

        recog_prediction, recog_hidden = self.recog_sequence_head(recog_contextual_feature.contiguous(), recog_target,
                                                                  is_train=False, return_hidden=True)

        # remove last hidden EOS to be compatible with bieo labels.
        recog_hidden = recog_hidden[:, :-1, :]

        info_feat_list.append(recog_hidden)

        text = self.recog_sequence_head.get_pred_text(recog_prediction, self.test_cfg.batch_max_length)
        result['texts'] = text

        # multimodal context module
        multimodal_context, _, _ = self.infor_context_module(info_feat_list,
                                                              pos_feat=_bboxes,
                                                              img_metas=img_metas,
                                                              info_labels=None,
                                                              bieo_labels=None)

        if self.with_infor_node_cls_head:
            cls_pred = self.infor_node_cls_head(multimodal_context)
            result['bboxes_labels_pred'] = cls_pred[0].cpu().numpy()

        # lstm + crf if required
        if self.with_infor_sequence_module and self.with_infor_ner_head:
            valid_multimodal_context = []
            for idx, _ in enumerate(_bboxes):
                valid_length = _bboxes[idx].size(0)
                valid_multimodal_context.append(multimodal_context[idx, :valid_length, :])
            valid_multimodal_context = torch.cat(valid_multimodal_context, 0)[:, None, :].repeat(1,
                                                                                                 recog_hidden.size(
                                                                                                     1), 1)

            fused_features = torch.cat((recog_hidden, valid_multimodal_context), -1)

            fused_features = self.infor_sequence_module(fused_features)
            outputs = self.infor_ner_head(fused_features)

            tags = self.infor_ner_head.get_predict(outputs)

            result['bboxes_bieo_labels_pred'] = tags[0].cpu().numpy()

        return [result]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Forward aug_test. Not implemented.
        """
        raise NotImplementedError
