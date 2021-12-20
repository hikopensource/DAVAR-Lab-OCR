"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    grid_category_head.py
# Abstract       :    Classification for each grid

# Current Version:    1.0.0
# Date           :    2021-03-19
######################################################################################################
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmcv.cnn import normal_init, ConvModule
from mmcv.runner import auto_fp16
from mmdet.models.builder import build_loss, HEADS


@HEADS.register_module()
class GridCategoryHead(nn.Module):
    """ Implement of category generation, refer to SOLOV2 [1].

    Ref: [1] SOLOv2: Dynamic, Faster and Stronger, NeurIPS-20
             <https://arxiv.org/abs/2003.10152>`_
    """

    def __init__(self,
                 num_grids,
                 sigma=0.2,
                 sample_point=20,
                 featmap_indices=(0, 1, 2, 3),
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=2,
                 stacked_convs=4,
                 loss_category=None):
        """
        Args:
            num_grids (list(int)): grid numbers in different feature map level
            sigma (float): GT shrink parameter
            sample_point (int): sample bboxes number
            featmap_indices (tuple(int)): feature maps levels
            in_channels (int): input feature map channels
            conv_out_channels (int): output feature map channels
            num_classes (int): category numbers
            stacked_convs (int): number of stack-convolutions
            loss_category (dict): loss function for grid category.
        """

        super().__init__()
        self.sigma = sigma
        self.sample_point = sample_point
        self.num_grids = num_grids
        self.featmap_indices = featmap_indices
        self.stacked_convs = stacked_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.category_out_channels = self.num_classes - 1
        if loss_category is not None:
            self.loss_category = build_loss(loss_category)
        else:
            self.loss_category = None
        self.cate_convs = nn.ModuleList()
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        for i in range(stacked_convs):
            chn = self.in_channels if i == 0 else self.conv_out_channels
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.conv_out_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.cate_out = nn.Conv2d(self.conv_out_channels,
                                  self.category_out_channels, 3, padding=1)

    def init_weights(self):
        """ Weight Initialization """
        for cate_conv in self.cate_convs:
            normal_init(cate_conv.conv, std=0.01)

    def forward_single(self, feats, num_grid):
        """ Forward computation in single level.

        Args:
            feats (Tensor): input feature map, in shape of [B, C, H, W]
            num_grid (int): split numbers S

        Returns:
            Tensor: output feature map, in shape of [B, S^2]
        """

        for idx in range(self.stacked_convs):
            if idx == 0:
                feats = F.interpolate(feats, size=num_grid, mode='bilinear')
            feats = self.cate_convs[idx](feats)
        feats = self.cate_out(feats)
        return feats

    @auto_fp16()
    def forward(self, feats):
        """ Forward computation in multiple levels.Refer to TextSnake [1] and Text Perceptron [2]
            Ref: [1] TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes. ECCV-18
                 [2] Text Perceptron: Toward End-to-End Arbitrary-shaped Scene Text Spotting. AAAI-20

        Args:
            feats (list(Tensor)): input feature maps, in shape of [B, C, H, W]

        Returns:
            list(Tensor): output feature maps, in shape of [B, S^2]
        """

        preds = []
        for i in range(len(self.featmap_indices)):
            pred = self.forward_single(feats[i], self.num_grids[i])
            preds.append(pred)
        return preds

    def _center_region_generate(self, bboxes):
        """ Generate the center region of text instances.

        Args:
            bboxes (list(list(float)): polygon bounding boxes for text instances.

        Returns:
            list(list(int)): center regions (polygon contours) of text instances.
        Returns:
            int: minimum edges of polygons
        """

        def point_distance(point_a, point_b):
            """ Calculate distance between to points.

            Args:
                point_a (list(float)): coordinate of the first point, [x, y]
                point_b (list(float)): coordinate of the second point, [x, y]

            Returns:
                float: distance between the two points.
            """

            return math.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)

        def dot_product(point_a, point_b, point_c, point_d):
            """ Calculate dot product for two vectors.

            Args:
                point_a (list(float)): start point of vector 1.
                point_b (list(float)): end point of vector 1.
                point_c (list(float)): start point of vector 2.
                point_d (list(float)): end point of vector 2.

            Returns:
                float: dot product of vector 1 and vector 2
            """

            return (point_b[0] - point_a[0]) * (point_d[1] - point_c[1]) - \
            (point_d[0] - point_c[0]) * (point_b[1] - point_a[1])

        def get_angle(point_a, point_b, point_c):
            """ Calculate angle of point_a->point_c->point_b

            Args:
                point_a (list(float)): start point
                point_b (list(float)): middle point
                point_c (list(float)): end point

            Returns:
                float: angle degree in [0, 360]
            """

            theta = math.atan2(point_a[0] - point_c[0], point_a[1] - point_c[1]) - math.atan2(point_b[0] - point_c[0],
                                                                                              point_b[1] - point_c[1])
            if theta > math.pi:
                theta -= 2 * math.pi
            if theta < - math.pi:
                theta += 2 * math.pi
            theta = theta * 180.0 / math.pi
            if theta < 0:
                theta = - theta
            if dot_product(point_a, point_c, point_c, point_b) < 0:
                theta = 360 - theta
            return theta

        def get_quad_index(bboxes):
            """ Estimate the corner points indexes and to make the top-left point as the index 0.
                For vertical instances, make the right-left point as the index 0.
                e.g., for quadrangle, the order is top-lef, top-right, bottom-right, bottom-left, respectively.

            Args:
                bboxes (list(list(float)): points of the polygon boxes. [ [x1, y1], ..., [xn, yn]]

            Returns:
                list(int): re-ordered corner points indexes.
            """

            # 4-point quadrangle shapes
            if bboxes.shape[0] == 4:
                tmp = np.zeros(4)
                tmp[0] = 0
                tmp[1] = 1
                tmp[2] = 2
                tmp[3] = 3

                # For vertical instances, make top-right point as the first point.
                if point_distance(bboxes[0], bboxes[3]) > 2 * point_distance(bboxes[0], bboxes[1]):
                    tmp[0] = 1
                    tmp[1] = 2
                    tmp[2] = 3
                    tmp[3] = 0
                return tmp
            angles = np.zeros(bboxes.shape[0])

            # The neighbor boundaries of head and tail boundary are nearly parrallel
            for i in range(bboxes.shape[0]):
                angle1 = get_angle(bboxes[i - 1], bboxes[(i + 1) % bboxes.shape[0]], bboxes[i])
                angle2 = get_angle(bboxes[i-2], bboxes[i], bboxes[i-1])
                angles[i] = abs(angle1 + angle2 - 180.0)
            tmp_index = 1
            ret = np.zeros(4)
            index = np.argsort(angles)
            while abs(index[0] - index[tmp_index]) == 1 or abs(index[0] - index[tmp_index]) == bboxes.shape[0] - 1:
                tmp_index += 1
            if index[0] < index[tmp_index]:
                ret[0] = index[0]
                ret[1] = (index[tmp_index] - 1 + bboxes.shape[0]) % bboxes.shape[0]
                ret[2] = index[tmp_index]
                ret[3] = (index[0] - 1 + bboxes.shape[0]) % bboxes.shape[0]
            else:
                ret[0] = index[tmp_index]
                ret[1] = (index[0] - 1 + bboxes.shape[0]) % bboxes.shape[0]
                ret[2] = index[0]
                ret[3] = (index[tmp_index] - 1 + bboxes.shape[0]) % bboxes.shape[0]
            return ret

        def get_sample_point(bboxes, sample_point_number):
            """ Averagely sample key points on top & bottom boundary of the polygon contour.

            Args:
                bboxes (list(list(int)): polygon bounding boxes of text instances.
                sample_point_number (int): point numbers in top / bottom boundary, which will generate 2*M points.

            Returns:
                list(list(int)): sampled points, in shape of [N, 2*M]
            """

            distance = np.zeros(bboxes.shape[0])
            length = np.zeros(bboxes.shape[0])
            sample_point = np.zeros((sample_point_number, 2))
            # Calculate the distance between adjacent points
            for i in range(bboxes.shape[0]-1):
                distance[i+1] = point_distance(bboxes[i], bboxes[i+1])
            # Calculate the distance between the 0-th point and the i-th point
            for i in range(1, bboxes.shape[0]):
                length[i] = length[i-1] + distance[i]
            # Calculate the average distance between adjacent points
            avg_distance = np.sum(distance) / (sample_point_number - 1)
            # Averagely sample points along the polygon
            for i in range(sample_point_number-1):
                cur_pos = avg_distance * i
                for j in range(bboxes.shape[0]-1):
                    if length[j] <= cur_pos < length[j+1]:
                        sample_point[i] = (bboxes[j + 1] - bboxes[j]) * (cur_pos - length[j]) / (
                                    length[j + 1] - length[j]) + bboxes[j]
            sample_point[-1] = bboxes[-1]
            return sample_point

        def get_center_poly(top_sample_point, down_sample_point):
            """ Calculate the center line based on the left&right boundary points

            Args:
                top_sample_point (np.ndarray): sample point number in top boundary
                down_sample_point (np.ndarray): sample point number in bottom boundary

            Returns:
                list(list(int)): centerline polygons
            """

            # Calculate the center point
            center_sample_point = (top_sample_point + down_sample_point) / 2
            distance = np.zeros(center_sample_point.shape[0])
            length = np.zeros(center_sample_point.shape[0])
            # Calculate the distance between adjacent center points
            for i in range(center_sample_point.shape[0]-1):
                distance[i+1] = point_distance(center_sample_point[i],
                                               center_sample_point[i+1])
            # Calculate the distance between the 0-th center point and the i-th center point
            for i in range(1, center_sample_point.shape[0]):
                length[i] = length[i-1] + distance[i]
            # Calculate the limits of the center point
            left_distance = 0.5 * (1 - self.sigma) * np.sum(distance)
            right_distance = 0.5 * (1 + self.sigma) * np.sum(distance)
            final_center_line = []
            # Calculate the center line
            for i in range(center_sample_point.shape[0] - 1):
                if length[i+1] > left_distance and length[i] < right_distance:
                    if len(final_center_line) == 0:
                        final_center_line.append(center_sample_point[i])
                    final_center_line.append(center_sample_point[i+1])
            return final_center_line

        bboxes = np.array(bboxes).reshape(-1, 2)

        # Estimate the corner points indexes
        quad_index = get_quad_index(bboxes).astype(int)
        if quad_index[0] > quad_index[1]:
            quad_index[1] += len(bboxes)
        if quad_index[2] > quad_index[3]:
            quad_index[3] += len(bboxes)

        # Calculate the boundary points based on the corner points indexes
        top_sample_point = []
        down_sample_point = []
        for i in range(quad_index[0], quad_index[1] + 1):
            top_sample_point.append(bboxes[i % len(bboxes)])
        for i in range(quad_index[2], quad_index[3] + 1):
            down_sample_point.append(bboxes[i % len(bboxes)])

        # Calculate the length of the shortest side of the polygon
        min_length = min(point_distance(bboxes[quad_index[0] % len(bboxes)],
                                    bboxes[quad_index[3] % len(bboxes)]),
                         point_distance(bboxes[quad_index[1] % len(bboxes)],
                                    bboxes[quad_index[2] % len(bboxes)]))
        top_sample_point = np.array(top_sample_point)
        down_sample_point = np.array(down_sample_point)[::-1]

        # Averagely sample key points on the boundary of the polygon contour
        top_sample_point = get_sample_point(top_sample_point, self.sample_point)
        down_sample_point = get_sample_point(down_sample_point, self.sample_point)

        # Calculate the center line based on the boundary points
        center_line = get_center_poly(top_sample_point, down_sample_point)
        center_line = np.array(center_line).astype(int)
        return center_line, min_length

    def _get_target_single(self,
                           gt_poly_bboxes,
                           feat_size,
                           num_grid,
                           stride,
                           device='cuda'
                           ):
        """ Generating the mapping of gt_bboxes and its corresponding grid

        Args:
            gt_poly_bboxes (list(float): polygon bounding boxes for text instances, in shape of [K, L]
            feat_size (tuple(int)): feature map shape
            num_grid (int): split number
            stride (int): feature map stride
            device (str): running device type

        Returns:
            Tensor: matched bboxes, a binary mask in of shape [B, S^2]
        """

        batch, _, height, width = feat_size
        matched_bboxes = torch.zeros([batch, num_grid**2],
                                     dtype=torch.long, device=device)
        for batch_id in range(batch):
            batch_bboxes = gt_poly_bboxes[batch_id]
            for idx, bboxes in enumerate(batch_bboxes):
                # Calculate the centerline coordinates and the shortest side length
                center_line, min_length = self._center_region_generate(bboxes)

                # Scale coordinates according to the downsampling factor
                center_line = (center_line / float(stride)).astype(int)
                min_length = int(min_length * self.sigma / float(stride))
                min_length = min_length if min_length > 0 else 1
                poly_mask = np.zeros((height, width), dtype=np.uint8)

                # Fill poly_mask according to the centerline
                for i in range(len(center_line)-1):
                    cv2.line(poly_mask, tuple(center_line[i]),
                             tuple(center_line[i+1]), 1, min_length)

                # Calculate the valid grid according to poly_mask
                mask_coord_y, mask_coord_x = np.where(poly_mask == 1)
                mask_coord_x = (mask_coord_x * num_grid / width).astype(int)
                mask_coord_y = (mask_coord_y * num_grid / height).astype(int)
                mask_coord = mask_coord_y * num_grid + mask_coord_x
                unique_coord = np.unique(mask_coord)
                matched_bboxes[batch_id, unique_coord] = idx + 1
        return matched_bboxes

    def get_target(self, feats, gt_poly_bboxes):
        """ Generating the mapping of gt_bboxes and its corresponding grid

        Args:
           gt_poly_bboxes (list(list(float)):  polygon bounding boxes for text instances, in shape of [K, L]

        Returns:
           list(Tensor): matched bboxes, a binary mask in of shape [B, S^2]
        """

        cate_targets = []
        for i, stride_idx in enumerate(self.featmap_indices):
            stride = 4 * (2**stride_idx)
            target = self._get_target_single(
                gt_poly_bboxes,
                feats[i].shape,
                self.num_grids[i],
                stride,
                device=feats[i].device,
            )
            cate_targets.append(target)
        return cate_targets

    def loss(self, cate_preds, cate_targets):
        """ Loss computation

        Args:
            cate_preds (list(Tensor)): feature map predictions
            cate_targets (list(Tensor)): feature map targets

        Returns:
            dict: losses in a dict
        """

        loss = dict()
        for i, stride_idx in enumerate(self.featmap_indices):
            stride = 4 * (2 ** stride_idx)
            cate_pred = cate_preds[i]
            cate_pred = cate_pred.permute(0, 2, 3, 1).view(-1, self.category_out_channels)

            cate_target = cate_targets[i]
            cate_target = torch.ge(cate_target, 1).long()
            cate_target = cate_target.view(-1, self.category_out_channels)
            if self.loss_category is not None:
                loss_category = self.loss_category(cate_pred, cate_target)
                loss.update({"loss_category_{}x".format(stride):loss_category})
        return loss
