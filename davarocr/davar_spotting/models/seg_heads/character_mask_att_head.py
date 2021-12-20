"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    character_mask_att_head.py
# Abstract       :    Character Mask Attention predtion

# Current Version:    1.0.0
# Date           :    2021-03-19
######################################################################################################
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import build_loss, HEADS
from mmcv.cnn import normal_init
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, auto_fp16
import cv2
import numpy as np


@HEADS.register_module()
class CharacterMaskAttentionHead(nn.Module):
    """ Inplemenation of CMA in MANGO[1]. Dynamic convolution strategy refers to Solov2 [2].

    Ref: [1] MANGO: A Mask Attention Guided One-Staged Text Spotter. AAAI-21.
             <https://arxiv.org/abs/2012.04350>`_
         [2] SOLOv2: Dynamic, Faster and Stronger, NeurIPS-20
             <https://arxiv.org/abs/2003.10152>`_
    """
    def __init__(self,
                 in_channels,
                 conv_out_channels,
                 num_grids,
                 stacked_convs=4,
                 text_max_length=25,
                 featmap_indices=(0, 1, 2, 3),
                 loss_char_mask_att=None,
                 ):
        """
        Args:
            in_channels (int): input feature map channel
            conv_out_channels (int): output feature map channel
            num_grids (list(int)): split img into S*S grids. List for 4x 8x 16x .. feature maps. e.g. [40, 40, 40, 40]
            stacked_convs (int): stacked convolution layers number
            text_max_length (int): the max length of recognition words.
            featmap_indices (list(int)): selected feature map scales.
            loss_char_mask_att (dict): loss function for CMA supervision, which is requried in pretraining stage.
        """

        super().__init__()
        assert len(num_grids) == len(featmap_indices)
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.text_max_length = text_max_length
        self.stacked_convs = stacked_convs
        self.fp16_enabled = False
        self.num_grids = num_grids
        self.featmap_indices = featmap_indices

        if loss_char_mask_att is not None:
            self.loss_char_mask_att = build_loss(loss_char_mask_att)
            self.loss_weight = loss_char_mask_att['loss_weight']
        else:
            self.loss_char_mask_att = None

        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.kernal_convs = nn.ModuleList()
        self.feature_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.conv_out_channels
            self.kernal_convs.append(
                ConvModule(
                    chn,
                    self.conv_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
            self.feature_convs.append(
                ConvModule(
                    chn,
                    self.conv_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        kernal_out_channels = self.conv_out_channels * self.text_max_length
        self.kernal_out = nn.Conv2d(self.conv_out_channels,
                                    kernal_out_channels,
                                    kernel_size=1,
                                    padding=0)
        self.feature_out = ConvModule(self.conv_out_channels,
                                      self.conv_out_channels,
                                      kernel_size=1,
                                      padding=0,
                                      norm_cfg=norm_cfg,
                                      bias=norm_cfg is None)

    def init_weights(self):
        """ Weight initialization. """
        for kernal_conv in self.kernal_convs:
            normal_init(kernal_conv.conv, std=0.01)
        for kernal_conv in self.feature_convs:
            normal_init(kernal_conv.conv, std=0.01)

    def forward_single(self, feats, matched_bboxes, num_grid):
        """ Forward of CMA in single level.

        Args:
            feats (Tensor): Input feature map, in shape of [B, C, H, W]
            matched_bboxes (Tensor): A tensor of shape [B, S^2] to indicate grid categories
            num_grid (int): An int number to indicate grid split numbers

        Returns:
            Tensor: in shape of [B, K, L, H, W]
        """

        kernal_feature = feats
        mask_feature = feats

        # Calculate x-axis and y-axis coordinate features
        x_range = torch.linspace(-1, 1, kernal_feature.shape[-1], device=kernal_feature.device)
        y_range = torch.linspace(-1, 1, kernal_feature.shape[-2], device=kernal_feature.device)
        y_coord, x_coord = torch.meshgrid(y_range, x_range)
        y_coord = y_coord.expand([kernal_feature.shape[0], 1, -1, -1])
        x_coord = x_coord.expand([kernal_feature.shape[0], 1, -1, -1])
        coord_feature = torch.cat([x_coord, y_coord], 1)

        # B x C x H x W -> B x (C+2) x H x W
        kernal_feature = torch.cat([kernal_feature, coord_feature], 1)
        # B x C x H x W -> B x (C+2) x H x W
        mask_feature = torch.cat([mask_feature, coord_feature], 1)

        # Generate dynamic convolution kernel
        for idx in range(self.stacked_convs):
            if idx == 0:
                # B x (C+2) x H x W -> B x (C+2) x S x S
                kernal_feature = F.interpolate(kernal_feature,size=num_grid, mode='bilinear')
            kernal_feature = self.kernal_convs[idx](kernal_feature)
            mask_feature = self.feature_convs[idx](mask_feature)
        kernal_feature = self.kernal_out(kernal_feature)  # B x CL x S x S
        mask_feature = self.feature_out(mask_feature)  # B x C x H x W
        batch, channel, height, width = mask_feature.shape

        # Calculate the number of valid grid
        max_category_num = torch.max(torch.sum(torch.ge(matched_bboxes, 1), dim=1))
        max_category_num = max(max_category_num, 1)

        # Calculate the location of valid grid
        _, indices = torch.topk(matched_bboxes, max_category_num)  # B x K
        indices = indices.unsqueeze(-1).expand(-1, -1, self.text_max_length * channel)  # B x K x CL
        kernal_feature = kernal_feature.permute(0, 2, 3, 1).contiguous().view(batch, num_grid**2, -1)  # B x SS x CL

        # Collect the values of valid grid
        kernal_feature = torch.gather(kernal_feature, 1, indices)  # B x K x CL

        # B x C x H x W -> BC x H x W -> 1 x BC x H x W
        mask_feature = mask_feature.contiguous().view(-1, height, width).unsqueeze(0)

        # B x K x CL -> BKL x C-> BKL x C x 1 x 1
        kernal_feature = kernal_feature.view(-1, channel).unsqueeze(-1).unsqueeze(-1)
        mask_pred = F.conv2d(mask_feature, kernal_feature, groups=batch).contiguous().view(batch, max_category_num,
                                                                                           self.text_max_length,
                                                                                           height,
                                                                                           width)  # B x K x L x H x W
        return mask_pred

    @auto_fp16()
    def forward(self, feats, matched_bboxes):
        """ Forward of CMA in multiple levels.

        Args:
            feats (list(Tensor)): Input feature maps, in shapes of [B, C, H, W]
            matched_bboxes (list(Tenor)): A tensor of shape [B, S^2] ot indicate grid categories

        Returns:
            list(Tensor): in shape of [B, K, L, H, W]
        """

        preds = []
        for i in range(len(self.featmap_indices)):
            pred = self.forward_single(feats[i], matched_bboxes[i],
                                       self.num_grids[i])
            preds.append(pred)
        return preds

    def get_target_single(self,
                          gt_cbboxes,
                          matched_bboxes,
                          feat_size,
                          stride,
                          device='cuda'
                          ):
        """ Ground-truth generated according to character level annotations in single level.

        Args:
            gt_cbboxes (list(list(list(float))): A variable length list [N, *, *].
                                                 e.g. [[[x1, y1, x2, y2, x3, y3, x4, y4],[],[]]...]
            matched_bboxes (Tensor): A tensor of shape [B, S^2] ot indicate grid categories
            feat_size (tuple): inpur feature map shape
            stride (int): An int number to indicate feature map stride
            device (str): computation device, default in 'cuda'

        Returns:
            Tensor:  ground-truth mask in single level, in shape of [B, K, L, H, W]
        Returns:
            Tensor:  channel-wised weight, in shape of [B, K, L]
        """

        batch, _, height, width = feat_size
        max_category_num = torch.max(torch.sum(torch.ge(matched_bboxes,1), dim=1))
        max_category_num = max(max_category_num, 1)
        gt_mask = torch.zeros([batch, max_category_num,self.text_max_length,
                               height, width], dtype=torch.uint8, device=device)
        mask_weight = torch.zeros([batch, max_category_num,
                      self.text_max_length], dtype=torch.float, device=device)
        values, _ = torch.topk(matched_bboxes, int(max_category_num))  # B x K
        for batch_id in range(batch):
            val = values[batch_id]
            gt_cbbox = gt_cbboxes[batch_id]
            for idx, cboxes in enumerate(gt_cbbox):
                # Calculate the valid grid corresponding to text instance
                indices = torch.where(val == idx + 1)[0]
                if len(indices) == 0:
                    continue
                # Fill gt mask according to the gt_cbboxes
                cboxes = cboxes.reshape(-1, 4, 2)
                cboxes_downsample = (cboxes / float(stride)).astype(int)
                str_length = cboxes.shape[0]
                target_mask = np.zeros((self.text_max_length, height, width), dtype=np.uint8)
                for c_id in range(str_length):
                    cbox = cboxes_downsample[c_id, :, :]   # N x 4 x 2
                    cv2.fillPoly(target_mask[c_id, :, :], [cbox], color=1)
                target_mask = torch.Tensor(target_mask)
                # Assign gt to the corresponding grid
                for ind in indices:
                    gt_mask[batch_id, ind, ...] = target_mask
                    mask_weight[batch_id, ind, :str_length+1] = 1
        return gt_mask, mask_weight

    def get_target(self, feats, gt_cbboxes, matched_bboxes):
        """ Ground-truth generated according to character level annotations in multiple levels.

        Args:
            feats (list(Tensor)): input feature maps, in shape of [B, C, H, W]
            gt_cbboxes (list(list(list(float))): A variable length list [N, *, *].
                                                 e.g. [[[x1, y1, x2, y2, x3, y3, x4, y4],[],[]]...]
            matched_bboxes (list(Tensor)): A tensor of shape [B, S^2] ot indicate grid categories

        Returns:
            list(tuple(Tensor)):  ground-truth mask in single level, in shape of [B, K, L, H, W] and
                                  channel-wised weight, in shape of [B, K, L]
        """

        mask_targets = []
        for i, stride_idx in enumerate(self.featmap_indices):
            stride = 4 * (2 ** stride_idx)
            target = self.get_target_single(
                gt_cbboxes,
                matched_bboxes[i],
                feats[i].shape,
                stride,
                device=feats[i].device
            )
            mask_targets.append(target)
        return mask_targets

    @force_fp32(apply_to=('mask_preds', ))
    def loss(self, mask_preds, mask_targets):
        """ Loss computation.

        Args:
            mask_preds (list(Tensor)): feature map predictions, in shape of [B, K, L,  H, W]
            mask_targets (list(Tensor)): feature map targets, in shape of [B, K, L]

        Returns:
            dict: losses in a dict.
        """

        loss = dict()
        for i, stride_idx in enumerate(self.featmap_indices):
            stride = 4 * (2 ** stride_idx)
            mask_pred = mask_preds[i]
            mask_pred = torch.sigmoid(mask_pred)
            _, _, length, height, width = mask_pred.shape
            gt_mask, mask_weight = mask_targets[i]

            mask_pred = mask_pred.view(-1, length, height, width)
            gt_mask = gt_mask.view(-1, length, height, width)
            mask_weight = mask_weight.view(-1, length).unsqueeze(-1).unsqueeze(-1)

            loss_mask_att = self.loss_char_mask_att(mask_pred, gt_mask, weight_in_channel=mask_weight)
            loss.update({"loss_cha_mask_att_{}x".format(stride):loss_mask_att})
        return loss
