"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    structures.py
# Abstract       :    BitmapMasks designed for LGPMA

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

import numpy as np
import torch
from mmcv.ops.roi_align import roi_align
from mmdet.core import BitmapMasks


class BitmapMasksTable(BitmapMasks):
    """Inherited from BitmapMasks. Modify the data type of mask to store pyramid mask
    """

    def __init__(self, masks, height, width):
        """
        Args:
            masks (ndarray): ndarray of masks in shape (N, H, W), where N is the number of objects.
            height (int): height of masks
            width (int): width of masks
        """

        super().__init__(
            masks=masks,
            height=height,
            width=width)

    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        device='cpu',
                        interpolation='bilinear'):
        """The only difference from the original function is that change resized mask from np.uint8 to np.float.

        Args:
            bboxes (Tensor): Bboxes in format [x1, y1, x2, y2], shape (N, 4)
            out_shape (tuple[int]): Target (h, w) of resized mask
            inds (ndarray): Indexes to assign masks to each bbox, shape (N,)
                and values should be between [0, num_masks - 1].
            device (str): Device of bboxes
            interpolation (str): See `mmcv.imresize`

        Return:
            BitmapMasksTable: the cropped and resized masks.
        """

        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(
            num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = torch.from_numpy(self.masks).to(device).index_select(
                0, inds).to(dtype=rois.dtype)
            targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape,
                                1.0, 0, 'avg', True).squeeze(1)
            resized_masks = targets.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)
