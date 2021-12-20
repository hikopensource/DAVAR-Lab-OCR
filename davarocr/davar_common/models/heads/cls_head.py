"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    cls_head.py
# Abstract       :    classification head.

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmdet.models.builder import HEADS, build_loss
from davarocr.davar_common.utils import get_root_logger


@HEADS.register_module()
class ClsHead(nn.Module):
    """Classification head.
    """
    def __init__(self,
                 input_size,
                 num_classes,
                 loss_cls=None):
        """
        Args:
            input_size (int): the dim of input features.
            num_classes (int): classes number.
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
            logger.info("Cls Head:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, img):
        """Forward implementation.

        Args:
            img(Tensor): input feature of shape [BxC]

        Returns:
            Tensor: output feature of shape [BxN], where N is num_classes
        """
        pred = self.fc_logits(img)
        return pred

    def get_target(self, gt_labels):
        """direct return gt labels.
        """
        return gt_labels

    def loss(self, pred, target, prefix=''):
        """ Loss computation

        Args:
            pred (list(Tensor)): prediction of shape [BxL(Optional, Length)xN]
            target (Tensor): groundtruth of shape [BxL(Optional, Length)]
        Returns:
            dict: containing classification loss.
        """
        loss = dict()
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)
        loss_key = prefix + 'loss_cls_ce'
        loss[loss_key] = self.loss_cls(pred, target)
        return loss
