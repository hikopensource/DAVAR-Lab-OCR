"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_roi_extractor.py
# Abstract       :    Extract RoI masking features from a single level feature map.

# Current Version:    1.0.0
# Date           :    2021-07-14
##################################################################################################
"""
import numpy as np

import torch
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor


@ROI_EXTRACTORS.register_module()
class MaskRoIExtractor(SingleRoIExtractor):
    """ Implementation of RoI masking feature extractor. """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        """
        Args:
            roi_layer (dict): Specify RoI layer type and arguments.
            out_channels (int): Output channels of RoI layers.
            featmap_strides (List[int]): Strides of input feature maps.
            finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        """

        super().__init__(roi_layer, out_channels, featmap_strides, finest_scale)

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, masks, roi_scale_factor=None):
        """ Forward computation.

        Args:
            feats (list(Tensor)): original feature maps, in shape of [B x C x H x W]
            rois (Tensor): region of interest, in shape of [num_roi x 5]
            masks (list(BitmapMasks)): the mask corresponding to each img.
            roi_scale_factor (tuple): scale factor that RoI will be multiplied by.

        Returns:
            Tensor: extract RoI masking feature maps, in shape of [num_roi x C x H x W]
        """

        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)

        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1).expand(*expand_dims).reshape(
                    roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois)
                roi_feats_t *= mask
                roi_feats += roi_feats_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.

        if masks is not None:
            left = 0
            right = 0
            output_size = self.roi_layers[0].output_size
            crop_masks = []
            for mask in masks:
                num = mask.masks.shape[0]
                right += num
                # Crop mask from gt_masks according to roi
                crop_mask = mask.crop_and_resize(rois[left:right, 1:], output_size, 
                    np.array(range(num)), device=rois.device)
                left += num
                crop_masks_t = torch.tensor(crop_mask.masks).to(roi_feats.device)
                crop_masks.append(crop_masks_t)
            crop_masks = torch.cat(crop_masks)
            crop_masks = crop_masks.unsqueeze(1)
            roi_feats = roi_feats * crop_masks.detach()
        return roi_feats
