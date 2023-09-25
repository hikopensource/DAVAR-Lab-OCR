"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ctunet.py
# Abstract       :    Dataset format used in CTUNet

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""
import numpy as np
import torch
import torch.nn.functional as F
from davarocr.davar_common.models.builder import build_connect
from davarocr.davar_common.models.builder import build_embedding
from davarocr.davar_spotting.models import SPOTTER
from davarocr.davar_spotting.models import TwoStageEndToEnd
from mmdet.core import bbox2roi
from mmdet.models import build_head
from mmdet.models import build_roi_extractor


@SPOTTER.register_module()
class CTUNet(TwoStageEndToEnd):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 infor_roi_extractor=None,
                 raw_text_embedding=None,
                 infor_context_module=None,
                 infor_node_cls_head=None,
                 infor_row_relation_head=None,
                 infor_col_relation_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 rcg_backbone=None,
                 rcg_neck=None,
                 rcg_transformation=None,
                 rcg_sequence_module=None,
                 rcg_roi_extractor=None,
                 rcg_sequence_head=None,
                 ):
        """
        Args:
            backbone (dict): backbone of the model (e.g. ResNet)
            neck (dict): neck of the model (e.g., FPN)
            rpn_head (dict): head for generate proposal (e.g. RPNHead)
            roi_head (dict): head for predict mask/box according to roi (e.g. StandardRoIHead)
            infor_roi_extractor(dict)：information extractor roi
            raw_text_embedding(dict)：original text_embedding config
            infor_context_module(dict): context module for IE (e.g. MultiModalContextModule)
            infor_node_cls_head(dict): node cls head for IE (e.g. ClsHead)
            infor_row_relation_head(dict): edge cls head in row direction (e.g. ClsHead)
            infor_col_relation_head(dict): edge cls head in column direction (e.g. ClsHead)
            train_cfg(dict): training parameters
            test_cfg(dict): testing parameters
            pretrained (str, optional): pretrained model
            rcg_backbone (dict): backbone of the recognation model (e.g. ResNet)
            rcg_neck (dict): neck of the recognation model (e.g. FPN)
            rcg_transformation (dict): recognation feature transformation module (e.g. TPS, STN)
            rcg_sequence_head (dict): recognition head (e.g., AttentionHead)
            rcg_roi_extractor (dict): head for extract region of interest (e.g. SingleRoIExtractor)
            rcg_sequence_module (dict): module for extract sequence relation (e.g. RNN / BiLSTM/ Transformer)
        """
        super(CTUNet, self).__init__(
            backbone=backbone,
            rcg_roi_extractor=rcg_roi_extractor,
            rcg_sequence_head=rcg_sequence_head,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            rcg_backbone=rcg_backbone,
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
        if infor_context_module is not None:
            self.infor_context_module = build_connect(infor_context_module)
        if infor_node_cls_head is not None:
            self.infor_node_cls_head = build_head(infor_node_cls_head)
        if infor_row_relation_head is not None:
            self.infor_row_relation_head = build_head(infor_row_relation_head)
        if infor_col_relation_head is not None:
            self.infor_col_relation_head = build_head(infor_col_relation_head)

        self.init_weights(pretrained=pretrained)

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
    def with_infor_row_relation_head(self):
        """

        Returns:
            Determine the model with the infor_row_relation_head or not
        """
        return hasattr(self, 'infor_row_relation_head') and self.infor_row_relation_head is not None

    @property
    def with_infor_col_relation_head(self):
        """

        Returns:
            Determine the model with the infor_col_relation_head or not
        """
        return hasattr(self, 'infor_col_relation_head') and self.infor_col_relation_head is not None

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        super().init_weights(pretrained)

        if self.with_raw_text_embedding:
            self.raw_text_embedding.init_weights(pretrained)
        if self.with_infor_context_module:
            self.infor_context_module.init_weights(pretrained)
        if self.with_infor_node_cls_head:
            self.infor_node_cls_head.init_weights(pretrained)
        if self.with_infor_row_relation_head:
            self.infor_row_relation_head.init_weights(pretrained)
        if self.with_infor_col_relation_head:
            self.infor_col_relation_head.init_weights(pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_texts,
                      gt_rowcols=None,
                      array_gt_texts=None,
                      gt_bieo_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      relations=None,
                      proposals=None,
                      **kwargs):
        """ Forward train process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_bboxes (list(Tensor): Tensor bboxes for each image, in [x_tl, y_tl, x_br, y_br] order.
            gt_labels (list(Tensor): category labels for each bbox
            gt_texts (list(list(string))): text transcription: e.g. ["abc", "bte",....]
            gt_rowcols (list(list(list(int): row and column numbers of each cells, e.g. [row_s, col_s, row_e, col_e]
            array_gt_texts (list(np.array)): text transcription after tokenization in np.array format.
            gt_bieo_labels (list(list(list(float))): character-level labels
            gt_bboxes_ignore (list(Tensor)): ignored bboxes
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            relations (list(Tensor)): relationship between cells.
            proposals (list(list(np.array))): proposals for detection:
            **kwargs: other parameters

        Returns:
            dict: all losses in a dict
        """
        losses = dict()

        info_labels = gt_labels

        # ===================== text detection branch ====================
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

        # ===================== Multimodal Fusion Branch=========================
        if sum([box.shape[0] for box in gt_bboxes]) == 0:
            # non positive gts in current batch, we just not train the multimodal fusion branch
            return losses

        info_feat_list = []

        # visual features
        rois = bbox2roi(gt_bboxes)
        infor_feats = self.infor_roi_extractor(x[:self.infor_roi_extractor.num_inputs], rois)
        info_feat_list.append(infor_feats)

        # original text embedding
        char_nums = [(text > 0).sum(-1).tolist() for text in array_gt_texts]
        gt_texts_tensor = torch.from_numpy(np.concatenate(array_gt_texts, axis=0)).long().to(x[0].device)
        recog_hidden = self.raw_text_embedding(gt_texts_tensor)
        info_feat_list.append(recog_hidden)

        # multimodal context module
        multimodal_context, batched_img_label, batched_img_bieo_label, bert_token_embeddings = \
            self.infor_context_module(info_feat_list,
                                      pos_feat=gt_bboxes,
                                      img_metas=img_metas,
                                      info_labels=info_labels,
                                      bieo_labels=gt_bieo_labels,
                                      gt_texts=gt_texts,
                                      char_nums=char_nums)

        # ===================== Relational Graph Construction ====================

        # node classification
        if self.with_infor_node_cls_head:
            cls_pred = self.infor_node_cls_head(multimodal_context)
            cls_pred = cls_pred.view(-1, cls_pred.size(-1))
            tmp_labels = torch.cat(batched_img_label, 0).view(-1)
            valid_mask = tmp_labels != 255
            info_loss = self.infor_node_cls_head.multi_loss(cls_pred[valid_mask],
                                                            tmp_labels[valid_mask],
                                                            prefix='infor_node_')
            losses.update(info_loss)

        # row relation linking classification
        if self.with_infor_row_relation_head:
            banch_length = multimodal_context.size()[1]
            # get adjacent matrix
            multimodal_context_1 = multimodal_context[:, :, None, :].repeat(1, 1, banch_length, 1)
            multimodal_context_2 = multimodal_context[:, None, :, :].repeat(1, banch_length, 1, 1)
            pairwise_context = torch.cat([multimodal_context_1, multimodal_context_2], -1)

            cls_pred = self.infor_row_relation_head(pairwise_context)

            # get labels
            tmp_labels = []
            tmp_masks = []
            for per_img_order_idx, per_img_order in enumerate(relations):
                tmp_batch_img_label = batched_img_label[per_img_order_idx]
                tmp_batch_img_label = tmp_batch_img_label[:, None, None].repeat(1, banch_length, 1) + \
                                      tmp_batch_img_label[None, :, None].repeat(banch_length, 1, 1)
                tmp_batch_img_label[tmp_batch_img_label >= 255] = 255
                tmp_batch_img_label[tmp_batch_img_label < 255] = 0
                if per_img_order:
                    relations_np = np.array(per_img_order)
                    for (x, y) in np.argwhere(relations_np == 2):
                        tmp_batch_img_label[x, y, 0] = 1
                tmp_labels.append(tmp_batch_img_label)

                # achieve the relationship mask
                per_img_masks = torch.zeros_like(tmp_batch_img_label)
                per_img_gt_rowcols = np.array(gt_rowcols[per_img_order_idx])
                for i, rowcol in enumerate(per_img_gt_rowcols):
                    indexr = \
                        np.where((per_img_gt_rowcols[:, 2] >= rowcol[0]) & (per_img_gt_rowcols[:, 0] <= rowcol[2]))[0]
                    indexc = \
                        np.where((per_img_gt_rowcols[:, 3] >= rowcol[1]) & (per_img_gt_rowcols[:, 1] <= rowcol[3]))[0]
                    per_img_masks[indexr, i, 0] = 1
                    per_img_masks[i, indexr, 0] = 1
                    per_img_masks[indexc, i, 0] = 1
                    per_img_masks[i, indexc, 0] = 1
                tmp_masks.append(per_img_masks)
            tmp_labels = torch.stack(tmp_labels, axis=0)
            tmp_masks = torch.stack(tmp_masks, axis=0)

            # calculate loss
            cls_pred = cls_pred.view(-1, cls_pred.size(-1))
            tmp_labels = tmp_labels.view(-1)

            # valid_mask = (tmp_labels != 255)
            tmp_masks = tmp_masks.view(-1)
            valid_mask = (tmp_labels != 255) & (tmp_masks != 0)

            info_loss = self.infor_row_relation_head.multi_loss(cls_pred[valid_mask], tmp_labels[valid_mask],
                                                                prefix='infor_row_relation_')
            losses.update(info_loss)

        # col relation linking classification
        if self.with_infor_col_relation_head:
            banch_length = multimodal_context.size()[1]
            # get adjacent matrix
            multimodal_context_1 = multimodal_context[:, :, None, :].repeat(1, 1, banch_length, 1)
            multimodal_context_2 = multimodal_context[:, None, :, :].repeat(1, banch_length, 1, 1)
            pairwise_context = torch.cat([multimodal_context_1, multimodal_context_2], -1)

            cls_pred = self.infor_col_relation_head(pairwise_context)

            # get labels
            tmp_labels = []
            tmp_masks = []
            for per_img_order_idx, per_img_order in enumerate(relations):
                tmp_batch_img_label = batched_img_label[per_img_order_idx]
                tmp_batch_img_label = tmp_batch_img_label[:, None, None].repeat(1, banch_length, 1) + \
                                      tmp_batch_img_label[None, :, None].repeat(banch_length, 1, 1)
                tmp_batch_img_label[tmp_batch_img_label >= 255] = 255
                tmp_batch_img_label[tmp_batch_img_label < 255] = 0
                if per_img_order:
                    relations_np = np.array(per_img_order)
                    for (x, y) in np.argwhere(relations_np == 1):
                        tmp_batch_img_label[x, y, 0] = 1
                tmp_labels.append(tmp_batch_img_label)

                # achieve the relationship mask
                per_img_masks = torch.ones_like(tmp_batch_img_label)
                per_img_gt_rowcols = np.array(gt_rowcols[per_img_order_idx])
                for i, rowcol in enumerate(per_img_gt_rowcols):
                    indexc = \
                        np.where((per_img_gt_rowcols[:, 3] < rowcol[1]) | (per_img_gt_rowcols[:, 1] > rowcol[3]))[0]
                    per_img_masks[indexc, i, 0] = 0
                    per_img_masks[i, indexc, 0] = 0
                tmp_masks.append(per_img_masks)
            tmp_labels = torch.stack(tmp_labels, axis=0)
            tmp_masks = torch.stack(tmp_masks, axis=0)

            # calculate loss
            cls_pred = cls_pred.view(-1, cls_pred.size(-1))
            tmp_labels = tmp_labels.view(-1)

            # valid_mask = (tmp_labels != 255)
            tmp_masks = tmp_masks.view(-1)
            valid_mask = (tmp_labels != 255) & (tmp_masks != 0)

            info_loss = self.infor_col_relation_head.multi_loss(cls_pred[valid_mask], tmp_labels[valid_mask],
                                                                prefix='infor_col_relation_')
            losses.update(info_loss)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_texts=None,
                    gt_rowcols=None,
                    array_gt_texts=None,
                    **kwargs):
        """ Forward test process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_bboxes (list(Tensor): Tensor bboxes for each image, in [x_tl, y_tl, x_br, y_br] order.
            gt_texts (list(list(str))): transcriptions for text recognition:
            gt_rowcols (list(list(list(int): row and column numbers of each cells, e.g. [row_s, col_s, row_e, col_e]
            array_gt_texts (list(np.array)): text transcription after tokenization in np.array format.

        Returns:
            dict: formated inference results
        """
        result = dict()
        # extract feats
        x = self.extract_feat(img)

        # ===================== Multimodal Fusion Branch=========================
        info_feat_list = []
        # visual features
        rois = bbox2roi(gt_bboxes[0])
        infor_feats = self.infor_roi_extractor(x[:self.infor_roi_extractor.num_inputs], rois.to(torch.float32))
        info_feat_list.append(infor_feats)

        # original text embedding
        char_nums = [(text > 0).sum(-1).tolist() for text in array_gt_texts[0]]
        gt_texts_tensor = torch.from_numpy(np.concatenate(array_gt_texts[0], axis=0)).long().to(x[0].device)

        recog_hidden = self.raw_text_embedding(gt_texts_tensor)
        info_feat_list.append(recog_hidden)

        # multimodal context module
        multimodal_context, batched_img_label, batched_img_bieo_label, bert_token_embeddings = \
            self.infor_context_module(info_feat_list,
                                      pos_feat=gt_bboxes[0],
                                      img_metas=img_metas,
                                      info_labels=None,
                                      bieo_labels=None,
                                      gt_texts=gt_texts[0],
                                      char_nums=char_nums)

        # ===================== Relational Graph Construction ====================
        # node classification
        if self.with_infor_node_cls_head:
            cls_pred = self.infor_node_cls_head(multimodal_context)
            result['bboxes_labels_pred'] = F.softmax(cls_pred[0], dim=-1).cpu().numpy()

        # row relation linking classification
        if self.with_infor_row_relation_head:
            banch_length = multimodal_context.size()[1]
            multimodal_context_1 = multimodal_context[:, :, None, :].repeat(1, 1, banch_length, 1)
            multimodal_context_2 = multimodal_context[:, None, :, :].repeat(1, banch_length, 1, 1)
            pairwise_context = torch.cat([multimodal_context_1, multimodal_context_2], -1)

            cls_pred = self.infor_row_relation_head(pairwise_context)

            # relationship mask
            imgs_masks = torch.zeros_like(cls_pred)
            for per_img_rowcols_idx, per_img_rowcols in enumerate(gt_rowcols[0]):
                assert per_img_rowcols_idx == 0
                per_img_rowcols_np = np.array(per_img_rowcols)
                for i, rowcol in enumerate(per_img_rowcols_np):
                    indexr = \
                        np.where((per_img_rowcols_np[:, 2] >= rowcol[0]) & (per_img_rowcols_np[:, 0] <= rowcol[2]))[0]
                    indexc = \
                        np.where((per_img_rowcols_np[:, 3] >= rowcol[1]) & (per_img_rowcols_np[:, 1] <= rowcol[3]))[0]
                    imgs_masks[per_img_rowcols_idx, indexr, i, 1] = 1
                    imgs_masks[per_img_rowcols_idx, i, indexr, 1] = 1
                    imgs_masks[per_img_rowcols_idx, indexc, i, 1] = 1
                    imgs_masks[per_img_rowcols_idx, i, indexc, 1] = 1
            imgs_masks[:, :, :, 1] = (imgs_masks[:, :, :, 1] - 1) * 10000
            result['bboxes_edges_pred_row'] = F.softmax(cls_pred[0] + imgs_masks[0], dim=-1).cpu().numpy()

        # col relation linking classification
        if self.with_infor_col_relation_head:
            banch_length = multimodal_context.size()[1]
            multimodal_context_1 = multimodal_context[:, :, None, :].repeat(1, 1, banch_length, 1)
            multimodal_context_2 = multimodal_context[:, None, :, :].repeat(1, banch_length, 1, 1)
            pairwise_context = torch.cat([multimodal_context_1, multimodal_context_2], -1)

            cls_pred = self.infor_col_relation_head(pairwise_context)

            # relationship mask
            imgs_masks = torch.zeros_like(cls_pred)
            for per_img_rowcols_idx, per_img_rowcols in enumerate(gt_rowcols[0]):
                assert per_img_rowcols_idx == 0
                per_img_rowcols_np = np.array(per_img_rowcols)
                for i, rowcol in enumerate(per_img_rowcols_np):
                    indexc = \
                        np.where((per_img_rowcols_np[:, 3] < rowcol[1]) | (per_img_rowcols_np[:, 1] > rowcol[3]))[0]
                    imgs_masks[per_img_rowcols_idx, indexc, i, 1] = -10000.0
                    imgs_masks[per_img_rowcols_idx, i, indexc, 1] = -10000.0

            result['bboxes_edges_pred_col'] = F.softmax(cls_pred[0] + imgs_masks[0], dim=-1).cpu().numpy()

        return [result]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Forward aug_test. Not implemented.
        """
        raise NotImplementedError
