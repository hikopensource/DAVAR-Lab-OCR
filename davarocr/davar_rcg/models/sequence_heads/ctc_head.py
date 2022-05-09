"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ctc_head.py
# Abstract       :     Implementations of the ctc prediction layer, loss calculation and result converter

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import torch
import torch.nn as nn

from mmdet.models.builder import HEADS
from mmdet.models.builder import build_loss

from davarocr.davar_common.core.builder import build_converter


@HEADS.register_module()
class CTCHead(nn.Module):
    """CTC Recognition Head proposed in Ref [1]

       Ref [1]: An End-to-End Trainable Neural Network for Image-based Sequence Recognition
                   and Its Application to Scene Text Recognition  TPAMI
    """
    def __init__(self,
                 input_size,
                 converter=dict(
                     type='CTCLabelConverter',
                     character='0123456789abcdefghijklmnopqrstuvwxyz'),
                 loss_ctc=dict(
                     type='CTCLoss',
                     zero_infinity=True,
                     blank=0,
                     reduction='mean',
                     loss_weight=1.0)
                 ):
        """
        Args:
            input_size (int): input feature dim
            loss_ctc (dict): loss function parameter
            converter (dict): converter parameter
        """

        super(CTCHead, self).__init__()
        self.input_size = input_size

        # build the converter
        self.converter = build_converter(converter)
        self.num_classes = len(self.converter.character) + 1  # num_classes：[blank] + characters
        self.fc_logits = nn.Linear(self.input_size, self.num_classes, bias=True)

        # build the loss
        self.loss_ctc = build_loss(loss_ctc)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        return

    def forward(self, x, text=None, is_train=None):
        """

        Args:
            x (Torch.Tensor): input feature
            text (Torch.Tensor): label information
            is_train (bool): whether is training state or test state

        Returns:
            Torch.Tensor: model output feature

        """
        pred = self.fc_logits(x)
        return pred

    def convert(self, text):
        """
        Args:
            text (str): text format information

        Returns:
            Torch.Tensor: label information after converter

        """
        return self.converter(text)

    def get_target(self, gt_texts):
        """
        Args:
            gt_texts (str): text format label information

        Returns:
            Torch.Tensor: vector transformed by text label
        Returns:
            Torch.Tensor: vector transformed by text length

        """

        if gt_texts is not None:
            # transfer the label to the model supervision format
            text, length = self.converter.encode(gt_texts)
            return text, length

        return None, None

    def loss(self, pred, target):
        """

        Args:
            pred (Torch.Tensor): model prediction
            target (Torch.Tensor): label information

        Returns:
            Torch.Tensor: model training loss

        """

        gt_text, text_length = target
        loss = dict()
        pred = pred.squeeze(1)
        prob = pred.log_softmax(2)
        prob_size = torch.IntTensor([prob.size(1)] * prob.size(0))
        prob = prob.permute(1, 0, 2)

        # prob: B, L, C
        # torch.backends.cudnn.enabled=False
        loss_ctc = self.loss_ctc(prob, gt_text.cuda(), prob_size, text_length)
        # torch.backends.cudnn.enabled=True
        loss['loss_ctc'] = loss_ctc
        return loss

    def get_pred_text(self, pred, batch_max_length):
        """

        Args:
            pred (Torch.Tensor): model output feature
            batch_max_length (int): max output text length

        Returns:
            list(str): true text format prediction transformed by the converter

        """

        pred = pred.squeeze(1)
        batch_size = pred.size(0)
        length_for_pred = torch.cuda.IntTensor([batch_max_length] * batch_size)

        pred = pred[:, :batch_max_length, :]
        _, preds_index = pred.max(2)
        preds_index = preds_index.contiguous().view(-1)

        # transfer the model prediction to text
        preds_str = self.converter.decode(preds_index, length_for_pred)

        return preds_str
