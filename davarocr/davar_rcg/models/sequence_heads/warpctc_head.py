"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    warpctc_head.py
# Abstract       :    Implementations of the warp-ctc prediction layer, loss calculation and result converter

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import logging

import torch
import torch.nn as nn
import torch.nn.init as init

from mmcv.runner import load_checkpoint

from mmdet.models.builder import HEADS
from mmdet.models.builder import build_loss

from davarocr.davar_common.core.builder import build_converter


@HEADS.register_module()
class WarpCTCHead(nn.Module):
    """WarpCTC Recognition Head"""
    def __init__(self,
                 input_size,
                 converter=dict(
                     type='CTCLabelConverter',
                     character='0123456789abcdefghijklmnopqrstuvwxyz'),
                 loss_ctc=dict(
                     type='WarpCTCLoss',
                     blank=0,
                     size_average=False,
                     length_average=False,
                     loss_weight=1.0),
                 use_1x1conv=False,
                 ):
        """
        Args:
            input_size (int): input feature dim
            loss_ctc (dict): loss function parameter
            converter (dict): converter parameter
            use_1x1conv (bool): whether to use 1*1 convolution
        """

        super(WarpCTCHead, self).__init__()
        self.input_size = input_size

        # build the converter
        self.converter = build_converter(converter)
        self.num_classes = len(self.converter.character)  # + 1 num_classes
        self.use_1x1conv = use_1x1conv

        # whether to use convolution or linear to realize classification
        if self.use_1x1conv:
            self.fc_logits = nn.Conv2d(self.input_size, self.num_classes,
                                       kernel_size=1, stride=1,
                                       padding=0, bias=False)
        else:
            self.fc_logits = nn.Linear(self.input_size, self.num_classes)

        # build the loss
        self.loss_ctc = build_loss(loss_ctc)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("WarpCTCHead:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, param in self.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)

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
        if len(pred.shape) == 4:
            if self.use_1x1conv:
                pred = pred.permute(0, 2, 3, 1)  # nchw -> nhwc
            pred = pred.squeeze(1)               # n h w c
        prob = pred.log_softmax(2)
        prob_size = torch.IntTensor([prob.size(1)] * prob.size(0))
        prob = prob.permute(1, 0, 2)

        # torch.backends.cudnn.enabled=False
        loss_ctc = self.loss_ctc(prob, gt_text, prob_size, text_length)
        # torch.backends.cudnn.enabled=True
        loss['loss_warpctc'] = loss_ctc
        return loss

    def get_pred_text(self, pred, batch_max_length, beam_search=False, beam_size=2):
        """

        Args:
            pred (Torch.Tensor): model output feature
            batch_max_length (int): max output text length
            beam_search (bool): whether to use beam search to decode
            beam_size (int): beam size

        Returns:
            list(str): true text format prediction transformed by the converter

        """

        if len(pred.shape) == 4:
            if self.use_1x1conv:
                pred = pred.permute(0, 2, 3, 1)  # nchw -> nhwc
            pred = pred.squeeze(1)

        # whether use beam search
        if beam_search:
            pred = pred.log_softmax(2)
            batch_size = pred.size(0)
            preds_str = list()
            for b in range(batch_size):
                # transfer the model prediction to text
                beam_result = self.converter.ctc_beam_search_decoder(
                    pred[b, :, :], beam_size=beam_size)
                preds_str.append(beam_result)

        else:
            batch_size = pred.size(0)
            batch_max_length = pred.size(1)
            length_for_pred = torch.cuda.IntTensor([batch_max_length] *
                                                   batch_size)

            _, preds_index = pred.max(2)
            preds_index = preds_index.contiguous().view(-1)

            # transfer the model prediction to text
            preds_str = self.converter.decode(preds_index,
                                              length_for_pred,
                                              get_before_decode=True)

        return preds_str
