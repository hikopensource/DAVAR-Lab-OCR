"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mango.py
# Abstract       :    The main pipeline definition of MANGO

# Current Version:    1.0.0
# Date           :    2021-03-19
######################################################################################################
"""
import torch
import torch.nn as nn
from mmdet.models.builder import build_backbone, build_neck, build_head
from davarocr.davar_common.models.builder import build_connect
from davarocr.davar_common.core.builder import build_postprocess

from ..builder import SPOTTER
from .base import BaseEndToEnd


@SPOTTER.register_module()
class MANGO(BaseEndToEnd):
    """ One stage recognition framework(used for MANGO[1])

    Ref: [1] MANGO: A Mask Attention Guided One-Staged Text Spotter. AAAI-21.
             <https://arxiv.org/abs/2012.04350>`_
    """

    def __init__(self,
                 backbone,
                 neck,
                 centerline_seg_head=None,
                 grid_category_head=None,
                 instance_mask_att_head=None,
                 multi_mask_att_head=None,
                 attention_fuse_module=None,
                 semance_module=None,
                 multi_recog_sequence_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """
        Args:
            backbone (dict): backbone of the model (e.g. ResNet)
            neck (dict): neck of the model  (e.g., FPN)
            centerline_seg_head (dict): head for centerline segmentation (e.g., CenterlineSegHead)
            grid_category_head (dict): head for grid category classification (e.g., GridCategoryHead)
            instance_mask_att_head (dict): Instance-level attention map head (e.g., InstanceMaskAttentionHead)
            multi_mask_att_head (dict): Character-level attention map head (e.g., CharacterMaskAttentionHead)
            attention_fuse_module (dict): attention fusion operation. (e.g., AttFuseModule)
            semance_module (dict): module for extract semantic relation (e.g., Cascade RNN / BiLSTM/ Transformer)
            multi_recog_sequence_head (dict): recognition head (e.g., MultiRecogSeqHead)
            train_cfg (dict): model training cfg parameter
            test_cfg (dict): model test cfg parameter
            pretrained (str, optional): model path of the pre_trained model
        """

        super().__init__()

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        self.centerline_seg_head = build_head(centerline_seg_head) if centerline_seg_head is not None else None
        self.grid_category_head = build_head(grid_category_head) if grid_category_head is not None else None
        self.instance_mask_att_head = build_head(instance_mask_att_head) if instance_mask_att_head is not None else None
        self.multi_mask_att_head = build_head(
            multi_mask_att_head) if multi_mask_att_head is not None else None
        self.attention_fuse_module = build_connect(attention_fuse_module) if attention_fuse_module is not None else None
        self.semance_module = build_connect(semance_module) if semance_module is not None else None
        self.multi_recog_sequence_head = build_head(
            multi_recog_sequence_head) if multi_recog_sequence_head is not None else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        if hasattr(self.test_cfg, 'postprocess'):
            self.post_processor = build_postprocess(self.test_cfg.postprocess)
        else:
            self.post_processor = None

    @property
    def with_centerline_seg_head(self):
        """
        Returns:
            bool: Determine the model is with the centerline_seg_head or not
        """

        return hasattr(self, 'centerline_seg_head') and self.centerline_seg_head is not None

    @property
    def with_grid_category_head(self):
        """
        Returns:
            bool: Determine the model is with the grid_category_head or not
        """

        return hasattr(self, 'grid_category_head') and self.grid_category_head is not None

    @property
    def with_instance_mask_att_head(self):
        """
        Returns:
            bool: Determine the model is with the instance_mask_att_head or not
        """

        return hasattr(self, 'instance_mask_att_head') and self.instance_mask_att_head is not None

    @property
    def with_multi_mask_att_head(self):
        """
        Returns:
            bool: Determine the model is with the multi_mask_att_head or not
        """

        return hasattr(self, 'multi_mask_att_head') and self.multi_mask_att_head is not None

    @property
    def with_attention_fuse_module(self):
        """
        Returns:
            bool: Determine the model is with the attention_fuse_module or not
        """

        return hasattr(self, 'attention_fuse_module') and self.attention_fuse_module is not None

    @property
    def with_semance_module(self):
        """
        Returns:
            bool: Determine the model is with the semance_module or not
        """

        return hasattr(self, 'semance_module') and self.semance_module is not None


    @property
    def with_multi_recog_sequence_head(self):
        """
        Returns:
            bool: Determine the model is with the multi_recog_sequence_head or not
        """

        return hasattr(self, 'multi_recog_sequence_head') and self.multi_recog_sequence_head is not None

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """

        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)

        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for conv in self.neck:
                    conv.init_weights()
            else:
                self.neck.init_weights()

        if self.with_centerline_seg_head:
            self.centerline_seg_head.init_weights()

        if self.with_grid_category_head:
            self.grid_category_head.init_weights()

        if self.with_instance_mask_att_head:
            self.instance_mask_att_head.init_weights()

        if self.with_multi_mask_att_head:
            self.multi_mask_att_head.init_weights()

        if self.with_attention_fuse_module:
            self.attention_fuse_module.init_weights()

        if self.with_semance_module:
            self.semance_module.init_weights()

        if self.with_multi_recog_sequence_head:
            self.multi_recog_sequence_head.init_weights()

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
                      gt_poly_bboxes=None,
                      gt_texts=None,
                      gt_cbboxes=None
                      ):
        """ Forward train process. Only 4x feature map is implemented.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_poly_bboxes (list(list(float)): polygon bounding boxes for text instances,:
                                               e.g. [[x1, y1, x2, y2, ....],....,[x_n1, y_n2,...]]
            gt_texts (list(string)): text transcriptio: e.g. ["abc", "bte",....]
            gt_cbboxes (list(list(list(float))): character-level bounding boxes for text instances:
                                                 e.g. [[[x1_1,y1_1,...,x1_8,y1_8],[x2_1, y2_1, ..],[]], [[],[],[]],....]
        Returns:
            dict: all losses in a dict
        """

        # Feature Extraction
        losses = dict()
        feats = self.extract_feat(img)

        assert len(feats) >= 1

        # Text centerline segmentation
        if self.centerline_seg_head is not None:
            seg_pred = self.centerline_seg_head(feats)
            seg_target = self.centerline_seg_head.get_target(feats, gt_poly_bboxes)
            seg_loss = self.centerline_seg_head.loss(seg_pred, seg_target)
            losses.update(seg_loss)

        # Grid category predict
        assert self.grid_category_head is not None
        grid_cate_target = self.grid_category_head.get_target(feats, gt_poly_bboxes)
        if self.grid_category_head.loss_category is not None:
            grid_category_pred = self.grid_category_head(feats)
            grid_category_loss = self.grid_category_head.loss(grid_category_pred, grid_cate_target)
            losses.update(grid_category_loss)

        # Instance mask attention
        if self.instance_mask_att_head is not None:
            instance_mask_att_pred = self.instance_mask_att_head(feats)
            instance_mask_att_target = self.instance_mask_att_head.get_target(feats, gt_poly_bboxes, grid_cate_target)
            instance_mask_att_loss = self.instance_mask_att_head.loss(instance_mask_att_pred, instance_mask_att_target)
            losses.update(instance_mask_att_loss)

        # Character mask attention
        assert self.multi_mask_att_head is not None
        character_mask_att_pred = self.multi_mask_att_head(feats, grid_cate_target)
        if self.multi_mask_att_head.loss_char_mask_att is not None:
            # This loss is used in pre-training stage
            character_mask_att_target = self.multi_mask_att_head.get_target(feats, gt_cbboxes, grid_cate_target)
            character_mask_att_loss = self.multi_mask_att_head.loss(character_mask_att_pred, character_mask_att_target)
            losses.update(character_mask_att_loss)

        # Text recognition
        if self.multi_recog_sequence_head is not None:
            assert self.attention_fuse_module is not None
            # Feature fusion
            att_feature = self.attention_fuse_module(feats, character_mask_att_pred)

            # Sequence's semantic feature extraction
            if self.semance_module is not None:
                att_feature = [self.semance_module(att_feature[0])]

            # Recognition head
            multi_recog_target = self.multi_recog_sequence_head.get_target(character_mask_att_pred, gt_texts,
                                                                           grid_cate_target)
            multi_recog_pred = self.multi_recog_sequence_head(att_feature, grid_cate_target)
            multi_recog_loss = self.multi_recog_sequence_head.loss(multi_recog_pred, multi_recog_target)
            losses.update(multi_recog_loss)

        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """ Forward test process
        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            rescale (boolean): if the image be re-scaled

        Returns:
            dict: formated inference results
        """

        # Feature extraction
        results = dict()

        feats = self.backbone(img)
        if self.neck is not None:
            feats = self.neck(feats)

        # Only predict in 4x feature map.
        feats = [feats[0]]

        # Centerline segmentation
        seg_preds = self.centerline_seg_head(feats)[0]  # 4x centerline featuremap, in shape of Tensor(B,1,H,W)
        results['seg_preds'] = seg_preds  # 4x feature, Tensor(B,1,H,W)

        cate_conf = self.grid_category_head(feats)[0]  # cate pred, in shape of Tensor(B, S^2)
        cate_conf = torch.sigmoid(cate_conf)

        bboxes_preds = []
        cate_preds = []
        cate_weights = []
        featmap_indices = self.centerline_seg_head.featmap_indices

        # BFS search
        for i in range(seg_preds.shape[0]):
            resize_factor = 4 * (2 ** featmap_indices[i])
            bbox_pred, cate_pred, weight_pred = self.post_processor.bfs_search(seg_preds[i], cate_conf[i])
            bbox_pred = bbox_pred * resize_factor
            bboxes_preds.append(bbox_pred)
            cate_preds.append(cate_pred)
            cate_weights.append(weight_pred)

        results['seg_preds'] = seg_preds  # 4x feature, Tensor(B, 1, H, W)
        results['cate_preds'] = cate_preds[0]  # 4x feature, Tensor(B, grid_num^2)
        results['bboxes_preds'] = bboxes_preds  # list(np.array(N, 8)), len=B
        results['cate_weights'] = cate_weights  # list(np.array(grid_num^2)), len=B

        if torch.sum(cate_preds[0]) == 0:
            # If there is no text region detected in grid category predictions
            results['text_preds'] = None
            results['character_mask_att_preds'] = None
        else:
            # Character Mask Attention predict
            character_mask_att_preds = self.multi_mask_att_head(feats, cate_preds)

            # Attention fusion
            att_feature = self.attention_fuse_module(feats, character_mask_att_preds)

            # Text recognition
            if self.semance_module is not None:
                att_feature = [self.semance_module(att_feature[0])]
            recog_preds = self.multi_recog_sequence_head(att_feature, cate_preds)
            text_preds = self.multi_recog_sequence_head.get_pred_text(recog_preds)

            results['text_preds'] = text_preds[0]  # 4x feature, list(B, N)
            results['character_mask_att_preds'] = character_mask_att_preds[0]  # 4x feature, Tensor(B,K,L,H,W)

        # Post-processing
        if self.post_processor is not None:
            results = self.post_processor.post_processing(results, img_metas)

        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
