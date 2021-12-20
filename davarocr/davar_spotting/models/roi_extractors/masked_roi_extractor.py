"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    masked_roi_extractor.py
# Abstract       :    Extract RoI masking features from a single level feature map.

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


@ROI_EXTRACTORS.register_module()
class MaskedRoIExtractor(nn.Module):
    """ Implementation of Masked RoI feature extractor, refer to PAN++ [1]

    Ref: [1] PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text. TPAMI-21.
             <https://arxiv.org/pdf/2105.00405>`_
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 featmap_strides,
                 output_size=(8, 32)):
        """ Network Initialization.

        Args:
            in_channels(int): Number of input channels
            out_channels(int): Number of output channels
            featmap_strides(list): head for loss calculation (e.g., TPHead)
            output_size(tuple): related parameters for training
        """
        super().__init__()
        self.featmap_strides = featmap_strides
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels

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
    def forward(self, feats, rois, masks):
        """ Masked RoI feature extract

        Args:
            feats (Tensor): input feature of shape (B, C, H, W)
            rois (Tensor): region of interest of shape (n, 5)
            masks (list(BitmapMasks)) : mask of each img

        Returns:
            Tensor: masked RoI feature
        """
        # Only using 4x feature
        feat = self.relu(self.bn(self.conv(feats[0])))

        scale_factor = 4
        batch_size, _, height, width = feat.size()
        pad = feat.new_tensor([-1, -1, 1, 1], dtype=torch.long).unsqueeze(0)
        
        # Rescale roi and mask
        bboxes = (rois[:, 1:] / scale_factor + pad).long()
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)].clamp(0, width)
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)].clamp(0, height)
        rescale_masks = [mask.rescale(1 / scale_factor) for mask in masks]

        roi_feats = []
        for batch_ind in range(batch_size):
            batch_feat = feat[batch_ind]
            batch_mask = rescale_masks[batch_ind].to_tensor(dtype=torch.float, device=batch_feat.device)
            ind = rois[:, 0] == batch_ind
            batch_bboxes = bboxes[ind]
            for i in range(len(batch_bboxes)):
                left, top, right, bottom = batch_bboxes[i]
                _, mask_hight, mask_width = batch_mask.size()
                right = min(right, mask_width)
                bottom = min(bottom, mask_hight)

                # Crop masked feature
                crop_feat = batch_feat[:, top:bottom, left:right] * batch_mask[i, top:bottom, left:right]
                _, crop_height, crop_width = crop_feat.size()

                # Rotate vertical instances
                if crop_height > crop_width * 1.5:
                    crop_feat = crop_feat.transpose(1, 2)
                if left >= right or top >= bottom:
                    crop_feat = np.zeros(self.output_size)
                else:
                    # Interpolate feature to fixed size
                    crop_feat = F.interpolate(crop_feat.unsqueeze(0), size=self.output_size, mode='bilinear')
                roi_feats.append(crop_feat)
        roi_feats = torch.cat(roi_feats)
        return roi_feats
