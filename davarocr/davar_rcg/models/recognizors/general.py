"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    general.py
# Abstract       :    Implementations of the General Recognizor Structure

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import torch
import numpy as np
from mmdet.models import builder

from davarocr.davar_common.models.builder import build_connect
from davarocr.davar_common.models.builder import build_transformation

from .base import BaseRecognizor
from ..builder import RECOGNIZORS
from .. import builder as recog_builder
from .test_mixins import TextRecognitionTestMixin


def word_acc(pred, target, topk=1):
    """
    Args:
        pred (tensor): model prediction
        target (tensor): label information
        topk (int): top k accuracy, default 1

    Returns:
        torch.Tensor: model prediction accuracy

    """
    assert topk == 1
    assert len(pred) == len(target) and len(target) > 0
    p_n = np.array(pred)
    t_n = np.array(target)
    acc = sum(p_n == t_n) * 1.0 / len(target)
    acc = torch.FloatTensor([acc]).cuda()
    return acc


@RECOGNIZORS.register_module()
class GeneralRecognizor(BaseRecognizor, TextRecognitionTestMixin):
    """General Recognizor class support CTC and Attn model"""
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
            train_cfg (mmcv.config): model training cfg parameter
            test_cfg (mmcv.config): model test cfg parameter
            pretrained (str): model path of the pre_trained model
        """
        super(GeneralRecognizor, self).__init__()

        # build the transformation network
        if transformation is not None:
            self.transformation = build_transformation(transformation)

        # build the backbone network
        self.backbone = builder.build_backbone(backbone)

        # build the neck network
        if neck is not None:
            self.neck = build_connect(neck)

        # build the sequence_module network
        if sequence_module is not None:
            self.sequence_module = build_connect(sequence_module)

        # build the head network
        self.sequence_head = builder.build_head(sequence_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        # default keep dim is 4
        self.keepdim_train = getattr(self.train_cfg, 'keep_dim', True)
        self.keepdim_test = getattr(self.test_cfg, 'keep_dim', True)

        # default keep dim is 4
        self.use_permute_train = getattr(self.train_cfg, 'use_permute', True)
        self.use_permute_test = getattr(self.test_cfg, 'use_permute', True)

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

        """
        super(GeneralRecognizor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.sequence_head.init_weights(pretrained=pretrained)

        if self.with_neck:
            self.neck.init_weights()

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
            img (tensor): training images
            gt_text (tensor): label information
            teacher_output (tensor): teacher model output
            **kwargs (None): back parameter

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
        visual_feature = self.backbone(encode_feature)  # B,C,H,W

        # neck network
        if self.with_neck:
            visual_feature = self.neck(visual_feature)

        batch, source_channels, source_height, source_width = visual_feature.size()
        visual_feature = visual_feature.view(batch, source_channels, 1, source_height*source_width)

        if self.keepdim_train:
            if self.use_permute_train:
                # B,C,1,W -> B,C,W -> B,1,L,C
                visual_feature = visual_feature.permute(0, 2, 3, 1)
        else:
            # B,C,1,W -> B,C,W -> B,L,C
            visual_feature = visual_feature.squeeze(2).permute(0, 2, 1)

        # sequence network
        if self.with_sequence_module:
            contextual_feature = self.sequence_module(visual_feature)
        else:
            contextual_feature = visual_feature

        # transfer the label information
        recog_target = self.sequence_head.get_target(gt_text)

        # prediction head
        # B,C,1,W -> B,C,W -> L,B,C
        prediction = self.sequence_head(contextual_feature.contiguous(),
                                        recog_target,
                                        is_train=True)

        loss_recog = self.sequence_head.loss(prediction,
                                             recog_target)

        losses.update(loss_recog)

        # show the training accuracy online
        # text = self.sequence_head.get_pred_text(prediction, self.test_cfg.batch_max_length)
        #
        # acc = word_acc(text, gt_text)
        # losses.update({'WordAcc': acc})  # too slow

        return losses

    def simple_test(self,
                    imgs,
                    gt_texts=None,
                    teach_mode=False,
                    **kwargs):
        """
        Args:
            imgs (tensor): training images
            gt_texts (tensor): label information
            teach_mode (tensor): whether to use teacher-student mode
            **kwargs (None): back parameter

        Returns:
            dict: result of the model inference text
        Returns:
            dict: result of the model inference text and probability

        """

        # transformation network
        if self.with_transformation:
            encode_feature = self.transformation(imgs)
        else:
            encode_feature = imgs

        # backbone network
        visual_feature = self.backbone(encode_feature)

        # neck network
        if self.with_neck:
            visual_feature = self.neck(visual_feature)

        batch, source_channels, source_height, source_width = visual_feature.size()
        visual_feature = visual_feature.view(batch, source_channels, 1, source_height*source_width)
        if self.keepdim_test:
            if self.use_permute_test:
                visual_feature = visual_feature.permute(0, 2, 3, 1)
        else:
            visual_feature = visual_feature.squeeze(2).permute(0, 2, 1)

        # sequence network
        if self.with_sequence_module:
            contextual_feature = self.sequence_module(visual_feature)
        else:
            contextual_feature = visual_feature

        # transfer the label information
        recog_target = self.sequence_head.get_target(gt_texts)

        # prediction head
        preds = self.sequence_head(contextual_feature.contiguous(),
                                   recog_target,
                                   is_train=False)

        text = self.sequence_head.get_pred_text(preds, self.test_cfg.batch_max_length)

        out_format = dict()
        if isinstance(text, list):
            out_format["text"] = text
        elif isinstance(text, tuple):
            out_format["text"] = [text[0]]
            out_format["prob"] = [text[1]]
        else:
            raise TypeError("Not supported data type in davarocr")

        return out_format
    

    def aug_test(self,
                 imgs,
                 gt_texts=None,
                 **kwargs):
        
        out_format = dict()
        result = self.aug_test_text_recognition(imgs, gt_texts, **kwargs)

        text = self.post_processing(result)

        out_format["text"] = [text]
        return out_format
