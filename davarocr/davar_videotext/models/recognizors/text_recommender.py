"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    yoro_rcg.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-06-05
##################################################################################################
"""

from mmdet.models import builder

from davarocr.davar_rcg.models.recognizors import GeneralRecognizor
from davarocr.davar_rcg.models.builder import RECOGNIZORS
from davarocr.davar_common.models import build_transformation
from davarocr.davar_common.models.builder import build_connect


@RECOGNIZORS.register_module()
class TextRecommender(GeneralRecognizor):
    """YORO Recognizor using Attn model"""
    def __init__(self,
                 backbone,
                 sequence_head,
                 neck=None,
                 transformation=None,
                 sequence_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """

        Args:
            backbone (dict): backbone parameter
            sequence_head (dict): sequence_head parameter
            neck (dict): neck parameter
            transformation (dict): transformation parameter
            train_cfg (dict): model training cfg parameter
            test_cfg (mmcv.config): model test cfg parameter
            pretrained (str): model path of the pre_trained model
        """
        super().__init__(backbone, sequence_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.head_cfg = {'head_cfg': self.train_cfg}

        # Build the transformation network
        if transformation is not None:
            self.transformation = build_transformation(transformation)

        # Build the backbone network
        self.backbone = builder.build_backbone(backbone)

        # Build the neck network
        if neck is not None:
            self.neck = build_connect(neck)

        # Build the sequence_module network
        if sequence_module is not None:
            self.sequence_module = build_connect(sequence_module)

        if self.train_cfg and self.train_cfg.get('fix_rcg', False):
            for p in self.parameters():
                p.requires_grad = False

        sequence_head = dict(sequence_head, **self.head_cfg)
        # Build the head network
        self.sequence_head = builder.build_head(sequence_head)

        self.init_weights(pretrained=pretrained)

        # Default keep dim is 4
        self.keepdim_train = getattr(self.train_cfg, 'keep_dim', True)
        self.keepdim_test = getattr(self.test_cfg, 'keep_dim', True)

        # Default keep dim is 4
        self.use_permute_train = getattr(self.train_cfg, 'use_permute', True)
        self.use_permute_test = getattr(self.test_cfg, 'use_permute', True)

    def forward_train(self,
                      img,
                      gt_text,
                      **kwargs):
        """
        Args:
            img (tensor): training images
            gt_text (tensor): label information
            **kwargs (None): back parameter

        Returns:
            dict: model training loss

        """

        if self.train_cfg and self.train_cfg.get('fix_rcg', False):
            self.backbone.eval()
        losses = dict()

        # Transformation network
        if self.with_transformation:
            x = self.transformation(img)
        else:
            x = img

        # Backbone network
        visual_feature, low_feature = self.backbone(x)  # B,C,H,W

        # Neck network
        if self.with_neck:
            visual_feature = self.neck(visual_feature)

        batch, source_channels, source_height, source_width = visual_feature.size()
        visual_feature = visual_feature.view(batch, source_channels, 1, source_height*source_width)

        if self.keepdim_train:
            if self.use_permute_train:
                visual_feature = visual_feature.permute(0, 2, 3, 1)
        else:
            visual_feature = visual_feature.squeeze(2).permute(0, 2, 1)

        # Sequence network
        if self.with_sequence_module:
            contextual_feature = self.sequence_module(visual_feature)
        else:
            contextual_feature = visual_feature

        # Transfer the label information
        gts = self.sequence_head.get_target([gt_text, kwargs['img_metas']])

        recog_target, _, _ = gts

        preds = self.sequence_head(contextual_feature.contiguous(),
                                        low_feature,
                                        recog_target,
                                        is_train=True)

        loss_recog = self.sequence_head.loss(preds,
                                             gts)
        losses.update(loss_recog)

        return losses

    def simple_test(self,
                    imgs,
                    gt_texts=None,
                    **kwargs):
        """
        Args:
            imgs (tensor): training images
            gt_texts (tensor): label information
            **kwargs (None): back parameter

        Returns:
            result of the model inference

        """
        batch_size = imgs.size(0)

        # Transformation network
        if self.with_transformation:
            x = self.transformation(imgs)
        else:
            x = imgs
        # Backbone network
        visual_feature, low_feature = self.backbone(x)

        # Neck network
        if self.with_neck:
            visual_feature = self.neck(visual_feature)

        batch, source_channels, source_height, source_width = visual_feature.size()
        visual_feature = visual_feature.view(batch, source_channels, 1, source_height*source_width)
        if self.keepdim_test:
            if self.use_permute_test:
                visual_feature = visual_feature.permute(0, 2, 3, 1)
        else:
            visual_feature = visual_feature.squeeze(2).permute(0, 2, 1)

        # Sequence network
        if self.with_sequence_module:
            contextual_feature = self.sequence_module(visual_feature)
        else:
            contextual_feature = visual_feature

        # Transfer the label information
        recog_target = self.sequence_head.get_target([gt_texts, kwargs['img_metas']])

        # Prediction head
        preds = self.sequence_head(contextual_feature.contiguous(),
                                   low_feature,
                                   recog_target,
                                   is_train=False)

        pred_texts, glimpse, track_feature, scores = preds
        text = self.sequence_head.get_pred_text(pred_texts, self.test_cfg.batch_max_length)
        out_format = dict()
        if isinstance(text, list):
            out_format["text"] = text
        elif isinstance(text, tuple):
            out_format["text"] = text[0]
            out_format["probs"] = text[1]
        else:
            raise TypeError("Not supported data type in davarocr")
        # Get output
        out_format['scores'] = scores
        out_format['glimpses'] = glimpse.reshape(batch_size, -1)
        out_format['track_feature'] = track_feature
        out_format['img_info'] = kwargs['img_metas']
        return out_format
