"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ace_head.py
# Abstract       :    Implementations of the ace prediction layer, loss calculation and result converter

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import logging

import torch
import torch.nn as nn
import torch.nn.init as init

from mmdet.models.builder import HEADS
from mmdet.models.builder import build_loss
from mmcv.runner import load_checkpoint

from davarocr.davar_common.core.builder import build_converter


@HEADS.register_module()
class ACEHead(nn.Module):
    """ACE Recognition Head proposed in Ref: [1]
      Ref: [1] Aggregation Cross-Entropy for Sequence Recognition. CVPR-2019
    """
    def __init__(self,
                 embed_size=512,
                 batch_max_length=25,
                 loss_ace=dict(
                     type="ACELoss",
                     character='0123456789abcdefghijklmnopqrstuvwxyz'
                 ),
                 converter=dict(
                     type='ACELabelConverter',
                     character='0123456789abcdefghijklmnopqrstuvwxyz', )
                 ):
        """
        Args:
            embed_size (int): feature embedding size
            loss_ace (dict): loss function parameter
            converter (dict): converter parameter
        """

        super(ACEHead, self).__init__()

        # build the loss
        self.loss_ace = build_loss(loss_ace)

        # build the converter
        self.converter = build_converter(converter)
        self.batch_max_length = batch_max_length
        self.num_class = len(self.converter.character)
        self.Prediction = nn.Linear(embed_size, self.num_class)
        self.soft_max = nn.Softmax(dim=-1)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("ACE_Head:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, param in self.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)

    def forward(self, encode_feature, recog_target=None, is_train=True):
        """
        Args:
            encode_feature (Torch.Tensor): encoded feature
            recog_target (Torch.Tensor): label information
            is_train (bool): whether is training state or test state

        Returns:
            Torch.Tensor: prediction results of each character category of the model

        """

        prediction = self.soft_max(self.Prediction(encode_feature))

        return prediction

    def convert(self, text):
        """
        Args:
            text (str): text format information

        Returns:
            Torch.Tensor: label information after converter

        """
        return self.converter(text)

    def loss(self, pred, target):
        """

        Args:
            pred (Torch.Tensor): model prediction
            target (Torch.Tensor): label information

        Returns:
            Torch.Tensor: model training loss

        """

        gt_label, _ = target

        loss = dict()

        loss_ace = self.loss_ace(pred, gt_label)

        # update the loss
        loss['loss_ace'] = loss_ace

        return loss

    def get_target(self, gt_texts):
        """
        Args:
            gt_texts (str): text format label information

        Returns:
            Torch.Tensor: vector transformed by text label
        Returns:
            Torch.Tensor: vector transformed text length
        """
        if gt_texts is not None:
            text, length = self.converter.encode(gt_texts, self.batch_max_length)
            return text, length

        return None, None

    def get_pred_text(self, preds, length):
        """

        Args:
            preds (Torch.Tensor): model output feature
            length (int): max output text length

        Returns:
            list(str): true text format prediction transformed by the converter

        """
        batch_size = preds.size()[0]
        _, preds_index = preds.max(2)

        length_for_pred = torch.cuda.IntTensor([length + 1] * batch_size)

        # transfer the model prediction to text
        preds_str = self.converter.decode(preds_index, length_for_pred)
        return preds_str
