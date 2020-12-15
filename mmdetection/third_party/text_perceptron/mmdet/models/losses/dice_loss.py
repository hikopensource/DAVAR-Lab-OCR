"""
#################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    dice_loss.py
# Abstract       :    implements of dice loss

# Current Version:    1.0.0
# Author         :    Liang Qiao
# Date           :    2020-05-31
#################################################################################################
"""

import torch
import torch.nn as nn

from mmdet.models.registry import LOSSES


def BinaryDiceLoss(predict, target, smooth, p, weight=None):
    """
    Description:
        Dice loss for binary classification
    """
    assert predict.shape[0] == target.shape[0]
    if weight is not None:
        predict = torch.mul(predict, weight)
        target = torch.mul(target, weight)

    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = torch.sum(torch.mul(predict, target))*2 + smooth
    den = torch.sum(predict.pow(p)+target.pow(p)) + smooth

    loss = 1 - num / den
    return loss


@LOSSES.register_module
class DiceLoss(nn.Module):
    """
    Descrition:
    	Dice loss for multi-class classification
    """
    def __init__(self,
                 smooth=1,
                 p=2,
                 loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                ):
        loss = self.loss_weight * self._multi_cls_loss(pred, target, weight)

        return loss

    def _multi_cls_loss(self, predict, target, weight):
        """
        Description:
            Dice loss for multi-class classification (as the expected value of multiple dices losses for binary classificaitions seperately)
 
        Arguments:
            predict:  feature map predictions, 
                      [N, num_classes, H, W]， where for num_classes classes，each contains a map of shape [H, W]
            target :  feature map ground-truth labels (one-hot encoding)
                      [N, num_classes, H, W]， where for num_classes classes，each contains a map of shape [H, W]
            weight :  mask (or weight) of feature map ground-truth labels,
                      no loss generates in the pixel if corresponding element of weight is 0的mask (weight)
        Returns:
            loss
        """
        assert predict.shape == target.shape
        if weight is not None:
            assert predict[0, 0].shape == weight[0].shape

        predict = torch.sigmoid(predict)
        total_loss = 0

        for i in range(target.shape[1]):
            dice_loss = BinaryDiceLoss(predict[:, i], target[:, i], self.smooth, self.p, weight=weight)
            total_loss += dice_loss
        return total_loss / target.shape[1]
