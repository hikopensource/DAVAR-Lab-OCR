"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    cross_entropy_loss.py
# Abstract       :    Implementation of the CrossEntropyLoss with ignore index

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""

import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class StandardCrossEntropyLoss(nn.Module):
    """Customized CrossEntropyLoss, add function to ignore index"""

    def __init__(self,
                 ignore_index=-100,
                 reduction='mean',
                 loss_weight=1.0):
        """

        Args:
            ignore_index (int): Specifies a target value does not contribute to the input gradient
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                           'none': no reduction will be applied,
                           'mean': the sum of the output will be divided by the number of elements in the output,
                           'sum': the output will be summed
            loss_weight (float): loss weight
        """
        super().__init__()

        assert reduction in ['mean', 'sum', 'none']

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                input_feature,
                target,
                weight=None):
        """
        Args:
            input_feature (Torch.tensor): model prediction
            target (Torch.tensor): label information
            weight (float|Torch.tensor): loss weight

        Returns:
            Torch.tensor: Standard Cross Entropy loss

        """

        # calculate the Standard Cross Entropy  loss
        loss_cls = self.loss_weight * F.cross_entropy(input_feature,
                                                      target,
                                                      weight=weight,
                                                      ignore_index=self.ignore_index,
                                                      reduction=self.reduction)
        return loss_cls
