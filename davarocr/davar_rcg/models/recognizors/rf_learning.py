"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    rf_learning.py
# Abstract       :    Implementations of the RF-leanring Recognizor Structure

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
from mmdet.models import builder

from davarocr.davar_common.models.builder import build_connect, build_transformation

from .base import BaseRecognizor
from ..builder import RECOGNIZORS


@RECOGNIZORS.register_module()
class RFLRecognizor(BaseRecognizor):
    """Reciprocal feature learning for scene text recognition"""
    def __init__(self,
                 backbone,
                 sequence_head,
                 counting_head,
                 neck=None,
                 neck_v2s=None,
                 neck_s2v=None,
                 transformation=None,
                 sequence_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 train_type="visual"):
        """

        Args:
            backbone (dict): backbone parameter
            sequence_head (dict): sequence_head parameter
            neck (dict): neck parameter
            neck_v2s (dict): visual to recognition feature strengthened neck parameter
            neck_s2v (dict): recognition to visual feature strengthened neck parameter
            transformation (dict): transformation parameter
            sequence_module (dict): sequence_module parameter
            train_cfg (mmcv.config): model training cfg parameter
            test_cfg (mmcv.config): model test cfg parameter
            pretrained (str): model path of the pre_trained model
            train_type (str): training type：
                                      1、"visual" - training visual counting branch
                                      2、"semantic" - training semantic recognition branch
                                      3、"total"  - training whole reciprocal feature learning
        """

        super(RFLRecognizor, self).__init__()

        # build the transformation network
        if transformation is not None:
            self.transformation = build_transformation(transformation)

        # build the backbone network
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = build_connect(neck)

        # build the v2s neck network
        if neck_v2s is not None:
            self.neck_v2s = build_connect(neck_v2s)

        # build the s2v neck network
        if neck_s2v is not None:
            self.neck_s2v = build_connect(neck_s2v)

        # build the sequence_module network
        if sequence_module is not None:
            self.sequence_module = build_connect(sequence_module)

        # build the head network
        self.sequence_head = builder.build_head(sequence_head)
        self.counting_head = builder.build_head(counting_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        # default keep dim is 4
        self.keepdim_train = getattr(self.train_cfg, 'keep_dim', True)
        self.keepdim_test = getattr(self.test_cfg, 'keep_dim', True)

        # default keep dim is 4
        self.use_permute_train = getattr(self.train_cfg, 'use_permute', True)
        self.use_permute_test = getattr(self.test_cfg, 'use_permute', True)
        self.train_type = train_type

    @property
    def with_transformation(self):
        """

        Returns:
            bool: Determine the model is with the transformation or not

        """
        return hasattr(self, 'transformation') and self.transformation is not None

    @property
    def with_neck(self):
        """

        Returns:
            bool: Determine the model is with the neck or not

        """
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_neck_v2s(self):
        """

        Returns:
            bool: Determine the model is with the neck_v2s or not

        """
        return hasattr(self, 'neck_v2s') and self.neck_v2s is not None

    @property
    def with_neck_s2v(self):
        """

        Returns:
            bool: Determine the model is with the neck_s2v or not

        """
        return hasattr(self, 'neck_s2v') and self.neck_s2v is not None

    @property
    def with_sequence_module(self):
        """
        Returns:
            bool: Determine the model is with the sequence_module or not

        """
        return hasattr(self, 'sequence_module') and self.sequence_module is not None

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        super(RFLRecognizor, self).init_weights(pretrained)

        self.backbone.init_weights(pretrained=pretrained)
        self.sequence_head.init_weights(pretrained=pretrained)
        self.counting_head.init_weights(pretrained=pretrained)

        if self.with_neck_v2s:
            self.neck_v2s.init_weights()

        if self.with_neck_s2v:
            self.neck_s2v.init_weights()

        if self.with_transformation:
            self.transformation.init_weights(pretrained=pretrained)

        if self.with_sequence_module:
            self.sequence_module.init_weights(pretrained=pretrained)

    def forward_train(self,
                      img,
                      gt_text,
                      teacher_output=None,
                      **kwargs):
        """
        Args:
            img (Torch.tensor): input feature
            gt_text (Torch.tensor): label information
            teacher_output (Torch.tensor): teacher model output
            **kwargs (None): backup parameter

        Returns:
            torch.Tensor: model training loss

        """

        losses = dict()

        # transformation network
        if self.with_transformation:
            encode_feature = self.transformation(img)
        else:
            encode_feature = img

        # backbone network
        visual_feature, rcg_feature = self.backbone(encode_feature)  # B,C,H,W

        batch, source_channels, source_height, source_width = visual_feature.size()
        visual_feature = visual_feature.view(batch, source_channels, 1, source_height*source_width)

        # neck network
        if self.with_neck_v2s:
            v_rcg_feature = rcg_feature * self.neck_v2s(visual_feature)
        else:
            v_rcg_feature = rcg_feature

        if self.with_neck_s2v:
            v_visual_feature = visual_feature + self.neck_s2v(rcg_feature)
        else:
            v_visual_feature = visual_feature

        batch, source_channels, source_height, source_width = v_rcg_feature.size()
        v_rcg_feature = v_rcg_feature.view(batch, source_channels, 1, source_height * source_width)

        if self.keepdim_train:
            if self.use_permute_train:
                # B,C,1,W -> B,C,W -> B,1,L,C
                v_rcg_feature = v_rcg_feature.permute(0, 2, 3, 1)
        else:
            # B,C,1,W -> B,C,W -> B,L,C
            v_rcg_feature = v_rcg_feature.squeeze(2).permute(0, 2, 1)

        # sequence network
        if self.with_sequence_module:
            contextual_feature = self.sequence_module(v_rcg_feature)
        else:
            contextual_feature = v_rcg_feature

        # transfer the label information
        recog_target = self.sequence_head.get_target(gt_text)
        visual_target = self.counting_head.get_target(gt_text)

        # prediction head
        # B,C,1,W -> B,C,W -> L,B,C
        prediction = self.sequence_head(contextual_feature.contiguous(),
                                        recog_target,
                                        is_train=True)

        prediction_visual = self.counting_head(v_visual_feature.contiguous())

        # whether to use teacher-student
        if teacher_output is None:
            loss_recog = self.sequence_head.loss(prediction,
                                                 recog_target)

            loss_count = self.counting_head.loss(prediction_visual,
                                                 visual_target)

        if self.train_type == "visual":
            losses.update(loss_count)
        elif self.train_type == "semantic":
            losses.update(loss_recog)
        else:
            losses.update(loss_count)
            losses.update(loss_recog)

        return losses

    def simple_test(self,
                    imgs,
                    gt_texts=None,
                    teach_mode=False,
                    **kwargs):
        """
        Args:
            imgs (Torch.tensor): training images
            gt_texts (Torch.tensor): label information
            teach_mode (Torch.tensor): whether to use teacher-student mode
            **kwargs (None): back parameter

        Returns:
            dict: model predicts text length
        Returns:
            dict: model predicts text
        Returns:
            dict: model predicts text length and text

        """

        # transformation network
        if self.with_transformation:
            encode_feature = self.transformation(imgs)
        else:
            encode_feature = imgs

        # backbone network
        visual_feature, rcg_feature = self.backbone(encode_feature)

        batch, source_channels, v_source_height, v_source_width = visual_feature.size()
        visual_feature = visual_feature.view(batch, source_channels, 1, v_source_height * v_source_width)

        # neck network
        if self.with_neck_v2s:
            v_rcg_feature = rcg_feature * self.neck_v2s(visual_feature)
        else:
            v_rcg_feature = rcg_feature

        if self.with_neck_s2v:
            v_visual_feature = visual_feature + self.neck_s2v(rcg_feature)
        else:
            v_visual_feature = visual_feature

        batch, source_channels, source_height, source_width = v_rcg_feature.size()
        v_rcg_feature = v_rcg_feature.view(batch, source_channels, 1, source_height * source_width)

        if self.keepdim_test:
            if self.use_permute_train:
                # B,C,1,W -> B,C,W -> B,1,L,C
                v_rcg_feature = v_rcg_feature.permute(0, 2, 3, 1)
        else:
            # B,C,1,W -> B,C,W -> B,L,C
            v_rcg_feature = v_rcg_feature.squeeze(2).permute(0, 2, 1)

        # sequence network
        if self.with_sequence_module:
            contextual_feature = self.sequence_module(v_rcg_feature)
        else:
            contextual_feature = v_rcg_feature

        # prediction head
        recog_target = self.sequence_head.get_target(gt_texts)

        preds = self.sequence_head(contextual_feature.contiguous(),
                                   recog_target,
                                   is_train=False)

        prediction_visual = self.counting_head(v_visual_feature.contiguous())

        out_format = dict()

        if self.train_type == "visual":
            length = self.counting_head.get_pred_length(prediction_visual)
            out_format["length"] = length
        else:
            if self.train_type == "semantic":
                text = self.sequence_head.get_pred_text(preds, self.test_cfg.batch_max_length)
            else:
                length = self.counting_head.get_pred_length(prediction_visual)
                text = self.sequence_head.get_pred_text(preds, self.test_cfg.batch_max_length)
                out_format["length"] = length

            if isinstance(text, list):
                out_format["text"] = text
            elif isinstance(text, tuple):
                out_format["text"] = text[0]
                out_format["prob"] = text[1]
            else:
                raise TypeError("Not supported data type in davarocr")
        return out_format
