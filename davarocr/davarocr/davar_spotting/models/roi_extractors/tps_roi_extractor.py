"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tps_roi_extractor.py
# Abstract       :    Extract RoI features according to tps rectification.

# Current Version:    1.0.0
# Date           :    2021-09-01
##################################################################################################
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from davarocr.davar_rcg.models.transformations.tps_transformation import GridGenerator
from davarocr.davar_spotting.utils.util_poly import get_sample_point, get_quad_index


@ROI_EXTRACTORS.register_module()
class TPSRoIExtractor(nn.Module):
    """ Implementation of tps based RoI feature extractor """

    def __init__(self,
                 in_channels,
                 out_channels,
                 featmap_strides,
                 point_num=14,
                 output_size=(8, 32)):
        """ Extractor initialization.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            featmap_strides (list[int]): strides of input feature maps
            point_num(int): the number of fiducial points in the boundaries
            output_size(tuple): output feature map size
        """
        super().__init__()
        self.featmap_strides = featmap_strides
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.point_num = point_num

        self.GridGenerator = GridGenerator(self.point_num, self.output_size)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    @property
    def num_inputs(self):
        """ Number of inputs of the features.

        Returns:
            int: number of input feature maps.
        """
        return len(self.featmap_strides)

    def init_weights(self):
        """ Parameters initialization """
        pass

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, fiducial_points):
        """ Feature rectification according to fiducial_points

        Args:
            feats (Tensor): input feature of shape [B x C x H x W]
            fiducial_points (list[np.array]): fiducial points of each text instance

        Returns:
            Tensor: rectification feature of shape [K x C x output_size]
        """
        roi_feats = []
        scale_factor = 4

        # only using 4x feature
        feats = self.relu(self.bn(self.conv(feats[0])))
        _, _, height, width = feats.size()

        for feat, points in zip(feats, fiducial_points):
            if len(points) == 0:
                continue
            points = torch.Tensor(points).cuda(device=feat.device)
            points = points / scale_factor
            for point in points:
                # Clip points
                point[:, 0] = torch.clip(point[:, 0], 0, width)
                point[:, 1] = torch.clip(point[:, 1], 0, height)

                # Caculate points boundary
                x1 = int(torch.min(point[:, 0]))
                x2 = int(torch.max(point[:, 0])) + 1
                y1 = int(torch.min(point[:, 1]))
                y2 = int(torch.max(point[:, 1])) + 1

                # Normalize points for tps
                point[:, 0] = 2 * (point[:, 0] - x1) / (x2 - x1) - 1
                point[:, 1] = 2 * (point[:, 1] - y1) / (y2 - y1) - 1

                # B x N (= output_size[0] x output_size[1]) x 2
                build_P_prime = self.GridGenerator.build_P_prime(point.unsqueeze(0))
                # B x output_size x 2
                build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0),
                                                               self.output_size[0],
                                                               self.output_size[1],
                                                               2])
                # Crop feature according to points boundary
                crop_feat = feat[:, y1:y2, x1:x2].unsqueeze(0)
                # B x C x output_size
                batch_I_r = F.grid_sample(crop_feat,
                                          build_P_prime_reshape,
                                          padding_mode='border')
                roi_feats.append(batch_I_r)
        roi_feats = torch.cat(roi_feats)
        return roi_feats

    def get_fiducial_points(self, imgs, polys):
        """ Compute tps fiducial points according to polygon contour.

        Args:
            imgs (Tensor): input image.
            polys (list(list(np.array))): poly boxes of text instances.

        Returns:
            list(np.array): tps fiducial points, in shape of [N, M, 2]
        """
        fiducial_points = []
        img_size = [img.size() for img in imgs]
        for batch_id, batch_bboxes in enumerate(polys):
            batch_fiducial_points = []
            _, height, width = img_size[batch_id]

            for box in batch_bboxes:
                box = np.array(box).reshape(-1, 2)

                # Estimate the corner points indexes
                quad_index = get_quad_index(box).astype(int)

                if quad_index[0] > quad_index[1]:
                    quad_index[1] += len(box)
                if quad_index[2] > quad_index[3]:
                    quad_index[3] += len(box)

                # Calculate the boundary points based on the corner points indexes
                top_sample_point = []
                down_sample_point = []
                for i in range(quad_index[0], quad_index[1] + 1):
                    top_sample_point.append(box[i % len(box)])
                for i in range(quad_index[2], quad_index[3] + 1):
                    down_sample_point.append(box[i % len(box)])

                top_sample_point = np.array(top_sample_point)
                down_sample_point = np.array(down_sample_point)[::-1]

                # Averagely sample key points on the boundary of the polygon contour
                top_sample_point = get_sample_point(top_sample_point, self.point_num // 2)
                down_sample_point = get_sample_point(down_sample_point, self.point_num // 2)

                fiducial_point = np.concatenate([top_sample_point, down_sample_point], axis=0)
                batch_fiducial_points.append(fiducial_point)

            if len(batch_bboxes) > 0:
                batch_fiducial_points = np.stack(batch_fiducial_points, axis=0)

            fiducial_points.append(batch_fiducial_points)
        return fiducial_points

    def rescale_fiducial_points(self, imgs, img_metas, fiducial_points):
        """ Rescale the fiducial points coordinates.

        Args:
            imgs (Tensor): input image.
            img_metas (dict): image meta-info.
            fiducial_points list(np.array): tps fiducial points.

        Returns:
            list(np.array): Rescaled points
        """
        normalized_fiducial_points = []
        for img, img_meta, point in zip(imgs, img_metas, fiducial_points):
            _, height, width = img.size()
            scale_factor = img_meta['scale_factor']
            if len(point) > 0:
                point = np.array(point, dtype=np.float).reshape(len(point), -1, 2)

                # Rescale
                point[:, :, 0] = point[:, :, 0] * scale_factor[0]
                point[:, :, 1] = point[:, :, 1] * scale_factor[1]

                # Change points order
                point_num = int(point.shape[1] / 2)
                point[:, point_num:, :] = point[:, point_num:, :][:, ::-1, :]
            normalized_fiducial_points.append(point)
        return normalized_fiducial_points
