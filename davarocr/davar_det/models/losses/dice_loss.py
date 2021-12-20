"""
#################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    dice_loss.py
# Abstract       :    Implements of dice loss. Refer to https://github.com/hubutui/DiceLoss-PyTorch

# Current Version:    1.0.0
# Date           :    2020-05-31
#################################################################################################
"""

import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES


def binaray_dice_loss(predict, target, smooth=1, p=2, weight=None):
    """Dice loss for binary classification

    Args:
        predict(Tensor): a tensor of shape [N, H, W]
        target(Tensor): a tensor of shape same with predict
        smooth(float): a float number to smooth loss, and avoid NaN error, default:1
        p(int): Denominator value, \sum{x^p}+\sum{y^p}, default:2
        weight: (Tensor): pixel-wised loss weight, the shape is [H, W]

    Returns:
        Tensor: loss tensor
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


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """Dice loss for multi-class classification. [1]

    Ref:  https://github.com/hubutui/DiceLoss-PyTorch
    """
    def __init__(self,
                 smooth=1,
                 p=2,
                 loss_weight=1.0,
                 use_sigmoid=False,
                 ):
        """ Initialization.

        Args:
            smooth(float): a float number to smooth loss, and avoid NaN error, default:1
            p(int): Denominator value, \sum{x^p}+\sum{y^p}, default:2
            loss_weight(float): loss weight
            use_sigmoid(bool): whether to conduct sigmoid operation on feature map
        """
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.loss_weight = loss_weight
        self.use_sigmoid = use_sigmoid

    def forward(self,
                pred,
                target,
                weight=None,
                weight_in_channel=None
                ):
        """ Multiply loss with loss_weight.

        Args:
            predict(Tensor): a tensor of shape [N, C, H, W]
            target(Tensor): a tensor of shape same with predict
            weight(Tensor): pixel-wised weight tensor, whose shape is [N, H, W]
            weight_in_channel(Tensor): channel-wised weight tensor, whose shape is [N, C]

        Returns:
            Tensor: loss tensor
        """
        loss = self.loss_weight * self._multi_cls_loss(pred, target, weight=weight, weight_in_channel=weight_in_channel)

        return loss

    def _multi_cls_loss(self, predict, target, weight=None, weight_in_channel=None):
        """Dice loss for multi-class classification (as the expected value of multiple dices
           losses for binary classificaitions seperately)

        Arg:
            predict(Tensor):  feature map predictions,
                              [N, num_classes, H, W], where for num_classes classes, each contains a map of shape [H, W]
            target(Tensor) :  feature map ground-truth labels (one-hot encoding)
                              [N, num_classes, H, W], where for num_classes classes, each contains a map of shape [H, W]
            weight(Tensor) :  [N, H, W], mask (or weight) of feature map ground-truth labels,
                             no loss generates in the pixel if corresponding element of weight is 0 mask (weight)
            weight_in_channel(Tensor): [N, num_classes], weight for channels

        Returns:
            loss tensor
        """
        assert predict.shape == target.shape

        if self.use_sigmoid:
            predict = predict.sigmoid()

        if weight is not None:
            assert predict[0, 0].shape == weight[0].shape
        if weight_in_channel is not None:
            predict = torch.mul(predict, weight_in_channel)
            target = torch.mul(target, weight_in_channel)

        total_loss = 0

        for i in range(target.shape[1]):
            dice_loss = binaray_dice_loss(predict[:, i], target[:, i], self.smooth, self.p, weight=weight)
            total_loss += dice_loss
        return total_loss / target.shape[1]
