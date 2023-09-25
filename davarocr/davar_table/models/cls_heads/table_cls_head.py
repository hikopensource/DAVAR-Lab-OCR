"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    table_cls_head.py
# Abstract       :    classification head using in CTUNet.

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmdet.models.builder import HEADS, build_loss
from davarocr.davar_common.utils import get_root_logger


@HEADS.register_module()
class TableClsHead(nn.Module):
    """Classification head used in table understanding.
    """

    def __init__(self,
                 input_size,
                 num_classes,
                 fc_out_channels=None,
                 num_fcs=0,
                 loss_cls=None):
        """
        Args:
            input_size (int): the dim of input features.
            num_classes (int): classes number.
            fc_out_channels (int): out channels of fc layers.
            num_fcs (int): numbers of fc layers.
            loss_cls (dict): loss config.
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc_out_channels = fc_out_channels
        self.num_fcs = num_fcs

        last_layer_dim = input_size
        self.extra_fc = nn.ModuleList()
        if self.fc_out_channels is not None and self.num_fcs > 0:
            for i in range(self.num_fcs):
                fc_in_channels = (
                    self.input_size if i == 0 else self.fc_out_channels)
                self.extra_fc.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels

        self.fc_logits = nn.Linear(last_layer_dim, self.num_classes)

        self.loss_cls = build_loss(loss_cls)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("TableCls Head:")
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
        for fc in self.extra_fc:
            img = self.relu(fc(img))
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

    def multi_loss(self, pred, target, mask=None, prefix=''):
        """Multi loss computation

        Args:
            pred (list(Tensor)): prediction of shape [BxL(Optional, Length)xN]
            target (Tensor): groundtruth of shape [BxL(Optional, Length)]
            mask: None
            prefix(str): loss type
        Returns:
            dict: containing classification loss.
        """
        loss = dict()
        loss_key = prefix + 'loss_cls_ce'
        loss[loss_key] = self.loss_cls(pred, target)
        return loss

    def get_predict(self, pred, target=None, mask=None):
        """ classification predict computation

        Args:
            pred (list(Tensor)): prediction of shape [BxL(Optional, Length)xN]
            target (Tensor): groundtruth of shape [BxL(Optional, Length)]
            mask: None
        Returns:
            list: Classification outputs.
        """
        lens = mask.t().sum(dim=0).int()
        predicts = []
        for i, length in enumerate(lens):
            per_sample = pred[i, :length]
            prev = torch.argmax(per_sample, dim=-1).cpu().numpy()
            predicts.append(prev)
        return predicts
