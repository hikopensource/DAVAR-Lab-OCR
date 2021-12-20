"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ner_head.py
# Abstract       :    NER head used in information extraction.

# Current Version:    1.0.0
# Date           :    2021-05-20
######################################################################################################
"""
from torch import nn
from mmcv.runner import load_checkpoint

from mmdet.models.builder import HEADS, build_loss
from davarocr.davar_common.utils import get_root_logger


@HEADS.register_module()
class NERHead(nn.Module):
    """NERHead implementation."""

    def __init__(self,
                 input_size,
                 num_classes,
                 loss_cls=None):
        """
        Args:
            input_size (int): input feature dim.
            num_classes (int): class number.
            loss_cls (dict): loss config.
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.fc_logits = nn.Linear(self.input_size, self.num_classes)

        self.loss_cls = build_loss(loss_cls)

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("NER Head:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input_feat):
        """ Forward computation

        Args:
            input_feat (Tensor): in shape of [B x L x M], where M is the dimension of features.
        Returns:
            Tensor: in shape of [B x L x D], where D is the num_classes.
        """
        pred = self.fc_logits(input_feat)
        return pred

    def get_target(self, gt_labels):
        """

        Returns:
            Tensor: gt labels, just return the inputs without any manipulation.
        """
        return gt_labels

    def loss(self, pred, target, mask=None, prefix=''):
        """ Compute the loss.

        Args:
            pred (Tensor): in shape of [B x L x D], where D is the num_classes.
            target (Tensor): in shape of [B x L], bieo label of per character/token.
            mask (Tensor): in shape of [B x L].
            prefix (str): loss prefix.

        Returns:
            dict: ner loss.
        """
        loss = dict()
        loss_key = prefix + 'ner_loss_cls_ce'
        loss[loss_key] = -self.loss_cls(emissions=pred, tags=target, mask=mask)
        return loss

    def get_predict(self, pred, mask=None):
        """ get the final predictions.

        Args:
            pred (Tensor): in shape of [B x L x D], where D is the num_classes.
            mask (Tensor): in shape of [B x L].

        Returns:
            Tensor: in shape of [1 x B x L], decoding labels of pred.
        """
        return self.loss_cls.decode(pred, mask=mask)
