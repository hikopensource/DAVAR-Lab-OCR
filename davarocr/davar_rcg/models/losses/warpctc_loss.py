"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    warpctc_loss.py
# Abstract       :    Implementations of the Baidu warp_ctc loss

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import torch.nn as nn

import warpctc_pytorch

from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class WarpCTCLoss(nn.Module):
    """Warp CTC Loss function"""
    def __init__(self,
                 blank=0,
                 size_average=False,
                 length_average=False,
                 loss_weight=1.0):
        """
        Args:
            blank (int): blank label, default 0
            size_average (bool): whether to norm the loss value with the batch size
            length_average (bool): whether to norm the loss value with length
            loss_weight (float): loss weight
        """
        super(WarpCTCLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = warpctc_pytorch.CTCLoss(blank=blank,
                                                 size_average=size_average,
                                                 length_average=length_average)

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
            Torch.Tensor: warp_ctc loss

        """

        # torch.backends.cudnn.enabled = False
        # calculate the warp_ctc loss
        loss_warpctc = self.loss_weight * self.criterion(log_probs,
                                                         targets.cpu(),
                                                         input_lengths.cpu(),
                                                         target_lengths.cpu())

        # torch.backends.cudnn.enabled = True
        return loss_warpctc.cuda()
