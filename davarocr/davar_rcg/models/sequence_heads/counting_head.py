"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    counting_head.py
# Abstract       :    Implementations of the rf-learning visual counting prediction layer,
                      loss calculation and result converter

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import logging

import torch.nn as nn
import torch.nn.init as init


from mmdet.models.builder import HEADS
from mmdet.models.builder import build_loss
from mmcv.runner import load_checkpoint

from davarocr.davar_common.core.builder import build_converter


@HEADS.register_module()
class CNTHead(nn.Module):
    """RF-learning Visual Counting Head proposed in Ref. [1]

    Ref: [1] Reciprocal Feature Learning via Explicit and Implicit Tasks in Scene Text Recognition. ICDAR-2021.
    """
    def __init__(self,
                 embed_size=512,
                 encode_length=26,
                 loss_count=dict(
                     type="MSELoss",
                     reduction='mean'
                 ),
                 converter=dict(
                     type='FemLabelConverter',
                     character='0123456789abcdefghijklmnopqrstuvwxyz', )
                 ):
        """
        Args:
            embed_size (int): embedding feature dim
            encode_length (int): backbone encodes feature width
            loss_count (dict): loss function parameter
            converter (dict): converter parameter
        """

        super(CNTHead, self).__init__()

        # build the loss
        self.loss_count = build_loss(loss_count)

        # build the converter
        self.converter_visual = build_converter(converter)

        self.num_class = len(self.converter_visual.character)

        self.Wv_fusion = nn.Linear(embed_size, embed_size, bias=False)
        self.Prediction_visual = nn.Linear(encode_length * embed_size, self.num_class)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("CNTHead:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, param in self.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)

    def forward(self, visual_feature):
        """
        Args:
            visual_feature (Torch.Tensor): visual counting feature

        Returns:
            Torch.Tensor: prediction results of each character category of the model
        """

        batch, channels, _, _ = visual_feature.size()

        visual_feature = visual_feature.reshape(batch, channels, -1).permute(0, 2, 1)
        visual_feature_num = self.Wv_fusion(visual_feature)  # batch * 26 * 512

        # using visual feature directly calculate the text length
        visual_feature_num = visual_feature_num.reshape(batch, -1)
        prediction_visual = self.Prediction_visual(visual_feature_num)

        return prediction_visual

    def convert(self, text):
        """
        Args:
            text (str): text format information

        Returns:
            Torch.Tensor: label information after converter

        """
        return self.converter_visual(text)

    def loss(self, pred, target):
        """

        Args:
            pred (Torch.Tensor): model prediction
            target (Torch.Tensor): label information

        Returns:
            Torch.Tensor: model training loss

        """
        gt_label = target
        loss = dict()
        loss_count = self.loss_count(pred, gt_label.float())

        # update the loss
        loss['loss_counting'] = loss_count

        return loss

    def get_target(self, gt_texts):
        """
        Args:
            gt_texts (str): text format label information

        Returns:
            Torch.Tensor: vector transformed by text label and converter

        """
        if gt_texts is not None:
            gt_embed = self.converter_visual.encode(gt_texts)
            return gt_embed

        return None

    def get_pred_length(self, preds):
        """

        Args:
            preds (Torch.Tensor): model output feature

        Returns:
            list(int): true text length transformed by the converter

        """

        # transfer the model prediction to text length
        preds_length = self.converter_visual.decode(preds)
        return preds_length
