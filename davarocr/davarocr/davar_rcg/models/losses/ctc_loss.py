"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ctc_loss.py
# Abstract       :    Implementations of the Torch official CTC loss

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class CTCLoss(nn.Module):
    """Torch CTC loss"""
    def __init__(self,
                 zero_infinity=True,
                 blank=0,
                 reduction='mean',
                 loss_weight=1.0):
        """
        Args:
            zero_infinity (bool): Whether to zero infinite losses and the associated gradients
            blank (int): blank label, default 0
            reduction (str): Specifies the reduction to apply to the output, including ['mean', 'sum']
                            'mean': the sum of the output will be divided by the number of elements in the output,
                            'sum': the output will be summed
            loss_weight (float): loss weight
        """
        super(CTCLoss, self).__init__()

        assert reduction in ['mean', 'sum', 'none']

        self.zero_infinity = zero_infinity
        self.blank = blank
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.criterion = torch.nn.CTCLoss(zero_infinity=zero_infinity,
                                          blank=blank,
                                          reduction=reduction)

    def forward(self,
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                ):
        """
        Args:
            log_probs (Torch.Tensor): model prediction
            targets (Torch.Tensor): label information
            input_lengths (Torch.Tensor): model prediction length
            target_lengths (Torch.Tensor): label information length

        Returns:
            Torch.Tensor: ctc loss

        """
        # torch.backends.cudnn.enabled=False
        # calculate the CTC loss
        loss_ctc = self.loss_weight * self.criterion(log_probs,
                                                     targets.cpu(),
                                                     input_lengths.cpu(),
                                                     target_lengths.cpu())
        # torch.backends.cudnn.enabled=True
        return loss_ctc.cuda()
