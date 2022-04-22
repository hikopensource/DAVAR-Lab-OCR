"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    lpma_mask_head.py
# Abstract       :    The main pipeline definition of lpma_mask_head

# Current Version:    1.0.1
# Date           :    2022-03-09
# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

import numpy as np
import torch
from torch import nn
from mmcv.runner import force_fp32
from mmdet.core.mask import mask_target
from mmdet.models.roi_heads import FCNMaskHead
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from davarocr.davar_table.core import get_lpmasks

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


@HEADS.register_module()
class LPMAMaskHead(FCNMaskHead):
    """ Implementation of LPMA branch.

    Ref: Qiao L, Li Z, Cheng Z, et al. LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask
     Alignment[J]. arXiv preprint arXiv:2105.06224, 2021. (Accepted by ICDAR 2021, Best Industry Paper)

    """

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_lpma=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        """
        Args:
            num_convs(int): number of convolutional layers in mask head
            roi_feat_size(int): size of RoI features
            in_channels(int): input feature map channels
            conv_kernel_size(int): kernel size of convolution layer
            conv_out_channels(int): the number of channels of output feature maps
            num_classes(int): class number.
            class_agnostic(bool): Class agnostic regresion or not
            upsample_cfg(None|str): upsample method
            conv_cfg(None|dict): config dict for convolution layer
            norm_cfg(None|dict): config dict for normalization layer
            loss_mask(dict): loss config of aligned cell region mask segmentation
            loss_lpma(dict): loss config of local pyramid mask segmentation
        """

        super(LPMAMaskHead, self).__init__(
            num_convs=num_convs,
            roi_feat_size=roi_feat_size,
            in_channels=in_channels,
            conv_kernel_size=conv_kernel_size,
            conv_out_channels=conv_out_channels,
            num_classes=num_classes,
            class_agnostic=class_agnostic,
            upsample_cfg=upsample_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            loss_mask=loss_mask)

        self.loss_lpma = build_loss(loss_lpma)
        logits_in_channel, out_channels = self.conv_logits.in_channels, self.conv_logits.out_channels + 2
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg, gt_bboxes=None):
        """ Generating gt_mask for training(LPMA branch)

        Args:
            sampling_results (List[:obj:``SamplingResult``]): Sampler results for each image.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            gt_bboxes (list(Tensor): Tensor bboxes for each image, in [x_1, y_1, x_2, y_2] order.

        Returns:
            list(Tensor): [text region mask targets, lpma targets in horizontal, lpma targets in vertical].
        """

        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        mask_targets = []
        gt_lpma_hor, gt_lpma_ver = get_lpmasks(gt_masks, gt_bboxes)
        for masks in [gt_masks, gt_lpma_hor, gt_lpma_ver]:
            mask_targets.append(mask_target(pos_proposals, pos_assigned_gt_inds, masks, rcnn_train_cfg))

        return mask_targets

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        """ Compute the loss.

        Args:
            mask_pred (Tensor): mask predictions
            mask_targets(list(Tensor)): mask targets
            labels (Tensor): (mask_target(Tensor),mask_weights(Tensor),geo_bond_target(Tensor),geo_weights(Tensor))

        Returns:
            loss(dict): Data flow, updated loss['loss_mask'] and loss['loss_lpma'].
        """

        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
            loss_lpma = mask_pred.sum()
        else:
            # split lp mask pred and mask pred
            mask_pred, lpmask_pred = mask_pred[:, :-2, :, :], mask_pred[:, -2:, :, :]
            lpmask_targets = torch.stack([mask_targets[1], mask_targets[2]], 1)
            loss_lpma = 5 * self.loss_lpma(lpmask_pred, lpmask_targets)

            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets[0], torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets[0], labels)

        loss['loss_mask'] = loss_mask
        loss['loss_lpma'] = loss_lpma

        return loss

    def get_seg_lpmasks(self, mask_pred, det_bboxes, det_labels, ori_shape, scale_factor, rescale):
        """Get local masks from mask_pred and det_bboxes in testing.

        Args:
            mask_pred (Tensor): mask predictions
            det_bboxes (Tensor): bBox predictions in shape (n, 5)
            det_labels (Tensor): label predictions in shape (n, )
            ori_shape (tuple): original image shape
            scale_factor(float | list(float)): ratio of original feature map to original image
            rescale(boolean): if the image be re-scaled

        Returns:
            list(list(np.array)): prediction of aligned cells region mask and global pyramid mask.
                like:[[mask1, mask2, ....], [pyramid mask1, pyramid mask2, ...]]
        """

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)]  # BG is not included in num_classes
        hor_segms, ver_segms = [], []
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            if isinstance(scale_factor, float):
                img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
                img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                    np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        if torch.onnx.is_in_onnx_export():
            # TODO: Remove after F.grid_sample is supported.
            from torchvision.models.detection.roi_heads import paste_masks_in_image
            masks = paste_masks_in_image(mask_pred, bboxes, ori_shape[:2])
            return masks

        N = len(mask_pred)
        # The actual implementation split the input into chunks, and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with skip_empty=True,
            # so that it performs minimal number of operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks, but may have memory issue
            num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        # gan hor soft masks
        mask_pred_hor = mask_pred[:, :1, :, :]
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.uint8)

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred_hor[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)
            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            hor_segms.append(im_mask[i].detach().cpu().numpy())

        # gan ver soft masks
        mask_pred_ver = mask_pred[:, 1:, :, :]
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.uint8)

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred_ver[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)
            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            ver_segms.append(im_mask[i].detach().cpu().numpy())

        for i in range(N):
            cls_segms[labels[i]].append([hor_segms[i], ver_segms[i]])

        return cls_segms
