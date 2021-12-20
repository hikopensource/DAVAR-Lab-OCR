# -*- coding: utf-8 -*-
"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    spatial_tempo_east_head.py
# Abstract       :    east head with YORO spatial temporal aggregation method

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import numpy as np
import torch
from torch import nn
import mmcv

from mmdet.models.builder import HEADS
from davarocr.davar_det.models.seg_heads import EASTHead



@HEADS.register_module()
class SpatialTempoEASTHead(EASTHead):
    """ Spatial Temporal aggregation module with EAST head
    """
    def __init__(self, loss_seg, loss_reg, geometry='RBOX', window_size=5, head_cfg=None):
        """
        Args:
            loss_seg(dict): loss for score map training
            loss_reg(dict): loss for geo map trainning
            geometry(dict): geo map mode, in "RBOX" or "QUAD"
            window_size(int): the size of consecutive frames
            head_cfg(dict): head config
        """
        super().__init__(loss_seg, loss_reg, geometry)

        self.head_cfg = head_cfg

        if self.head_cfg and self.head_cfg.get('fix_backbone', False):
            for param in self.parameters():
                param.requires_grad = False

        self.trans_conv = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.trans_bn = nn.BatchNorm2d(32)
        self.trans_relu = nn.ReLU(inplace=True)
        self.eps = 1e-9
        self.window_size = window_size

    def forward(self, x, img_meta):
        """ Forward compute of YORO head

        Args:
            x(Tensor): input feature map in shape of window_size x 32 x H/4 x W/4
            img_meta(list): the img meta information contains flow
        Returns:
            Tensor: score map in shape of [B, 1, H/4, W/4]
        Returns:
            Tensor: geo map in shape of [1, 5, H/4. W/4] or [1, 8, H/4. H/4]
        """
        feature_warps = x[:self.window_size]

        # The anchor frame always be the centor one, like Frame(t - 1), Frame(t), Frame(t + 1).
        target_idx = (self.window_size - 1) // 2

        # Fetch optical flow data
        flows = img_meta[target_idx]['flow']

        # Warping adjacent frame feature map to anchor feature according to flow
        feature_warps = self.warp(feature_warps, flows)

        # The mask to suppress not correct warping results
        warp_score_text_pred = self.score_map_sigmoid(self.score_map(feature_warps))
        target_map = warp_score_text_pred[target_idx]
        mask_text = (target_map >= 0.3).float()

        # Calculate the aggregation weight
        feature_trans = feature_warps

        # Warping feature embedding
        feature_trans = self.trans_conv(feature_trans)
        feature_trans = self.trans_bn(feature_trans)
        feature_trans = self.trans_relu(feature_trans)

        # Normalize features
        feature_trans = feature_trans / (self.eps + torch.sqrt(torch.sum(feature_trans ** 2, dim=1, keepdim=True)))

        # Calculate similarity map for Ft+i and Ft, [N, C, H, W]
        sim_weights = feature_trans * feature_trans[target_idx].unsqueeze(0)

        # Calculate similarity map for Ft+i and Ft, [N, 1, H, W]
        sim_weights = sim_weights.sum(dim=1, keepdim=True)

        # Temporal aggregation operation
        text_agg_weights = (sim_weights * warp_score_text_pred) / (
                    self.eps + (torch.sum((sim_weights * warp_score_text_pred), dim=0, keepdim=True)))

        # Temporal aggregation across the consecutive frames
        aggregation_anchor_score_map = torch.sum(text_agg_weights * warp_score_text_pred, dim=0)

        # Mask operation
        aggregation_anchor_score_map = aggregation_anchor_score_map * mask_text

        aggregation_anchor_score_map = aggregation_anchor_score_map.unsqueeze(0)

        # We only calculate the anchor frame loss
        x_target = x[target_idx].unsqueeze(0)

        if self.geometry == 'RBOX':
            geo_map_pre = self.geo_map_sigmod(self.geo_map(x_target))
            angle_map_pre = self.angle_map_sigmod(self.angle_map(x_target))

            # Balance distance and angle
            geo_map = torch.cat((self._power_layer(geo_map_pre),
                                 self._power_layer(angle_map_pre, 1, 1.570796327, -0.7853981635)),
                                dim=1)
        else:
            geo_map = self.geo_map(x_target)

        score_map = aggregation_anchor_score_map
        geo_map = geo_map[0].unsqueeze(0)
        return score_map, geo_map

    def simple_test(self, x, img_meta):
        """Forward inference, At the inference time, we test the result video by video and frame by frame, that means
        we first input like: [frame1, frame2, frame3, frame4, frame5], output the frame3 result, then we input [frame2,
        frame3, frame4, frame5, frame6], output frame4 result. So this means we could save the previous feature to
        accelerate the inference time

        Args:
            x(Tensor): input feature
            img_meta(dict): image meta-info

        Returns:
            Tensor: predicted warping score_map in shape of [B, 1, H/4, W/4]

        Returns:
            Tensor: predicted geo_map in shape of [1, 5, H/4. W/4] or [1, 8, H/4. H/4]
        """
        target_idx = (self.window_size - 1) // 2

        # This means we are going to test the first frame in a video, we have no previous feature to use
        if img_meta[-1]['pre_features'] is None:
            feature_warps = x
            flows = img_meta[-1]['flow']

        # Use the previous feature
        else:
            x = torch.cat((img_meta[-1]['pre_features'], x), dim=0)
            feature_warps = x
            flows = img_meta[-1]['flow']

        # Feature warping
        feature_warps = self.warp(feature_warps, flows)

        # Warping feature embedding
        feature_trans = feature_warps
        feature_trans = self.trans_conv(feature_trans)
        feature_trans = self.trans_bn(feature_trans)
        feature_trans = self.trans_relu(feature_trans)

        # Calculate the mask
        warp_score_text_pred = self.score_map_sigmoid(self.score_map(feature_warps))
        target_map = warp_score_text_pred[target_idx]
        mask_text = (target_map >= 0.3).float()

        # Normalize features
        feature_trans = feature_trans / (self.eps + torch.sqrt(torch.sum(feature_trans ** 2, dim=1, keepdim=True)))

        # Calculate similarity map for Ft+i and Ft, [N, C, H, W]
        sim_weights = feature_trans * feature_trans[target_idx].unsqueeze(
            0)

        # Calculate similarity map for Ft+i and Ft, [N, 1, H, W]
        sim_weights = sim_weights.sum(dim=1, keepdim=True)

        # Temporal aggregation operation
        text_agg_weights = (sim_weights * warp_score_text_pred) / (
                self.eps + (torch.sum((sim_weights * warp_score_text_pred), dim=0, keepdim=True)))

        # Temporal aggregation across the consecutive frames
        aggregation_anchor_score_map = torch.sum(text_agg_weights * warp_score_text_pred, dim=0)

        # Multiply normalized binary mask
        aggregation_anchor_score_map = aggregation_anchor_score_map * mask_text

        aggregation_anchor_score_map = aggregation_anchor_score_map.unsqueeze(0)

        # We only calculate the anchor frame
        x_target = x[target_idx].unsqueeze(0)

        if self.geometry == 'RBOX':
            geo_map_pre = self.geo_map_sigmod(self.geo_map(x_target))
            angle_map_pre = self.angle_map_sigmod(self.angle_map(x_target))

            # Balance distance and angle
            geo_map = torch.cat((self._power_layer(geo_map_pre),
                                 self._power_layer(angle_map_pre, 1, 1.570796327, -0.7853981635)),
                                dim=1)
        else:
            geo_map = self.geo_map(x_target)

        return aggregation_anchor_score_map, geo_map

    def get_target(self, gt_masks):
        """ Get split targets from gt_masks

        Args:
            gt_masks(Tensor): in shape of [B, 14, H/4, W/4] or [B, 20, H/4, W/4], where
                     gt_mask:[:,0]    :  gt_score_map
                     gt_mask:[:,1]    :  gt_score_map_mask, 1 Care / 0 Not Care
                     gt_mask:[:,2:7]  or gt_mask[:, 2:10] :  gt_geo_map
                     gt_mask:[:,7:12]  or gt_mask[:, 10:18] :  gt_geo_map

        Returns:
            Tensor: score map, in shape of [B, 1, H/4, W/4]
        Returns:
            Tensor: score map mask, in shape of [B, H/4, W/4]
        Returns:
            Tensor: geo map, in shape of [B, 5, H/4, W/4] or [B, 8, H/4, W/4]
        Returns:
            Tensor: geo map weights, in shape of [B, 5, H/4, W/4] or [B, 8, H/4, W/4]
        """
        score_map = gt_masks[:, 0:1, :, :]
        score_map_masks = gt_masks[:, 1, :, :].float()
        if self.geometry == 'RBOX':
            geo_map = gt_masks[:, 2:7, :, :]
            geo_map_weights = gt_masks[:, 7:12, :, :].float()
        else:
            geo_map = gt_masks[:, 2:10, :, :]
            geo_map_weights = gt_masks[:, 10: 18, :, :].float()

        if self.training:

            # We only need the anchor frame, that means the center one
            target_idx = (self.window_size - 1) // 2

            # Fetch the anchor frame gt score map
            target_score_map = score_map[target_idx].unsqueeze(0)
            score_map = target_score_map

            # Fetch the anchor frame gt score map mask
            target_score_map_masks = score_map_masks[target_idx].unsqueeze(0)
            score_map_masks = target_score_map_masks

            # Fetch the anchor frame gt geo map
            target_geo_map = geo_map[target_idx].unsqueeze(0)
            geo_map = target_geo_map

            # Fetch the anchor frame gt geo map mask
            target_geo_map_weights = geo_map_weights[target_idx].unsqueeze(0)
            geo_map_weights = target_geo_map_weights

        return score_map, score_map_masks, geo_map, geo_map_weights

    def loss(self, mask_pred, mask_targets):
        """ Compute loss

        Args:
            mask_pred(Tuple(Tensor)):  score_map in shape of [B, 1, H/4, W/4],
                                       geo_map in shape of [B, 5, H/4. W/4] or [B, 8, H/4. H/4]
            mask_targets(Tuple(Tensor)): score map, in shape of [B, 1, H/4, W/4]
                                         score map mask, in shape of [B, H/4, W/4]
                                         geo map, in shape of [B, 5, H/4, W/4] or [B, 8, H/4, W/4]
                                         geo map weights, in shape of [B, 5, H/4, W/4] or [B, 8, H/4, W/4]

        Returns:
            dict: loss in a dict
        """
        warp_score_pred, reg_pred = mask_pred
        score_map, score_map_masks, geo_map, geo_map_weights = mask_targets
        loss = dict()

        # Score_map loss
        loss["loss_seg_text"] = self.loss_seg(warp_score_pred, score_map, weight=score_map_masks)

        # Geo_map loss
        loss["loss_reg"] = self.loss_reg(reg_pred, geo_map, weight=geo_map_weights)
        return loss

    def warp(self, x, flow):
        """Warping operation

        Args:
            x(Tensor): input feature map in shape of window_size x 32 x H/4 x W/4
            flow(numpy array): optical flow in shape of window_size - 1 x height x width x 2

        Returns:
            Tensor: warping feature.
        """
        batch_size, _, height, width = x.size()
        target_idx = (batch_size - 1) // 2

        # By default, the flow don't save the anchor to anchor flow, because it's always zero and waste of storage
        flow = np.insert(flow, target_idx, np.zeros_like(flow[0]),axis=0)
        flows = []

        # Reshape flow size to be same with x size
        for batch_idx in range(batch_size):
            cur_flow = flow[batch_idx]
            cur_flow, w_scale, h_scale = mmcv.imresize(cur_flow, (width, height), return_scale=True)
            cur_flow[:, :, 0] = cur_flow[:, :, 0] * w_scale
            cur_flow[:, :, 1] = cur_flow[:, :, 1] * h_scale
            flows.append(cur_flow)

        flow = np.stack(flows)
        flow = torch.from_numpy(flow).permute(0, 3, 1, 2).contiguous()

        # Mesh grid
        xx = torch.arange(0, width).view(1, -1).repeat(height, 1)
        yy = torch.arange(0, height).view(-1, 1).repeat(1, width)
        xx = xx.view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
        yy = yy.view(1, 1, height, width).repeat(batch_size, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            flow = flow.to(x.device)
            grid = grid.to(x.device)

        vgrid = grid + flow

        # Normalize, cause grid_sample method need
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(width - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(height - 1, 1) - 1.0
        vgrid = vgrid.float()
        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid)
        return output
