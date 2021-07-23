"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    att_fuse_module.py
# Abstract       :    Multiply CMA with original feature map.

# Current Version:    1.0.0
# Date           :    2021-03-19
##################################################################################################
"""
import torch.nn as nn
from mmcv.cnn import normal_init, ConvModule
from mmcv.runner import auto_fp16
from davarocr.davar_common.models.builder import CONNECTS


@CONNECTS.register_module()
class AttFuseModule(nn.Module):
    """ Implementation of Attention map fusion.in MANGO [1]

    Ref: [1] MANGO: A Mask Attention Guided One-Staged Text Spotter. AAAI-21.
             <https://arxiv.org/abs/2012.04350>`_
    """

    def __init__(self,
                 featmap_indices=(0, 1, 2, 3),
                 in_channels=256,
                 conv_out_channels=256,
                 stacked_convs=4,
                 ):
        """
        Args:
            featmap_strides (tuple(int)): the selected N x feature maps, e.g (0, 1, 2, 3)
            in_channels (int): input feature map channels, single-level feature considered.
            conv_out_channels (int): output feature map channels, single-level feature considered
            stacked_convs (int): stacked convolutions number.
        """

        super().__init__()
        self.featmap_indices = featmap_indices
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.stacked_convs = stacked_convs
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.recog_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.conv_out_channels
            self.recog_convs.append(
                ConvModule(chn, self.conv_out_channels, 3, stride=1, padding=1, norm_cfg=norm_cfg,
                           bias=norm_cfg is None))

    def init_weights(self):
        """ Weight initialization """
        for recog_conv in self.recog_convs:
            normal_init(recog_conv.conv, std=0.01)

    @auto_fp16()
    def forward(self, feats, multi_mask_att):
        """ Forward computation.
            B: Batch size, C: Input Channels, H: Feature map height,
            W: Feature map width, K: Select channels 0 < K < grids_nums.

        Args:
            feats (list(Tensor)): original feature maps, in shape of [B x C x H x W]
            multi_mask_att (list(Tensor)): character mask attention feature maps, in shape of [B x K x L x H x W]

        Returns:
            list(Tensor): fused feature maps, in shape of [BK x L x C]
        """

        preds = []
        for i in range(len(self.featmap_indices)):
            pred = self.forward_single(feats[i], multi_mask_att[i])
            preds.append(pred)
        return preds

    def forward_single(self, feats, mask_att):
        """ Apply attention to feature map in different feature levels.
            B: Batch size, C: Input Channels, H: Feature map height,
            W: Feature map width, K: Select channels 0< K < grids_nums.

        Args:
            feats (Tensor): A tensor of shape [B, C, H, W]
            mask_att (Tensor): A tensor of shape [B, K, L, H, W]

        Returns:
            Tensor: A tensor of shape [B, K, L, C]
        """

        for idx in range(self.stacked_convs):
            feats = self.recog_convs[idx](feats)
        batch, channel, _, _ = feats.size()
        _, k, length, _, _ = mask_att.size()  # B x K x L x H x W
        mask_att = mask_att.sigmoid()
        mask_att = mask_att.view(batch, k * length, -1)  # B x KL x HW
        feats = feats.view(batch, channel, -1).permute(0, 2, 1)  # B x HW x C
        fused_feature = mask_att.matmul(feats).contiguous().view(batch * k, length, channel)  # BK x L x C
        return fused_feature
