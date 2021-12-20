"""
#################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    triplet_loss.py
# Abstract       :    Implements of triplet loss.

# Current Version:    1.0.0
# Date           :    2020-06-08
#################################################################################################
"""

from torch import nn

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class TripletLoss(nn.Module):
    """Triplet loss for metric learning

    """
    def __init__(self,
                 margin=1.0,
                 p=2,
                 loss_weight=1.0,
                 reduction='mean'):
        """ Initialization.

        Args:
            margin(float): a margin distance between for anchor-positive and anchor-negative
            p(int): Denominator value, \sum{x^p}+\sum{y^p}, default:2
            loss_weight(float): loss weight
        """
        super().__init__()
        self.margin = margin
        self.p = p
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.TripletMarginLoss(margin=self.margin, p=self.p, reduction=self.reduction)

    def forward(self,
                anchor,
                positive,
                negative
                ):
        """ Multiply loss with loss_weight.

        Args:
            anchor(Tensor): a tensor of shape [N, C, H, W]
            positive(Tensor): a tensor of shape same with anchor
            negative(Tensor): a tensor of shape same with anchor

        Returns:
            Tensor: loss tensor
        """

        loss = self.loss_weight * self.loss(anchor, positive, negative)

        return loss
