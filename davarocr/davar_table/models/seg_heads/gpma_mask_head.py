"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    gpma_mask_head.py
# Abstract       :    The main pipeline definition of gpma_mask_head

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

import mmcv
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class GPMAMaskHead(nn.Module):
    """ Implementation of GPMA branch.

    Ref: Qiao L, Li Z, Cheng Z, et al. LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask
     Alignment[J]. arXiv preprint arXiv:2105.06224, 2021. (Accepted by ICDAR 2021, Best Industry Paper)

    """

    def __init__(self,
                 in_channels=256,
                 conv_out_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 num_classes=81,
                 loss_mask=None,
                 loss_reg=None,
                 upsample_method=None,
                 upsample_ratio=None
                 ):
        """
        Args:
            in_channels(int): input channel of GPMAMaskHead
            conv_out_channels(int): output channel of GPMAMaskHead
            conv_cfg(None|dict): config dict for convolution layer
            norm_cfg(None|dict): config dict for normalization layer
            num_classes(int): number of classes
            loss_mask(dict): loss config of aligned cell region mask segmentation
            loss_reg(dict): loss config of global pyramid mask segmentation
            upsample_method(None|str): upsample method
            upsample_ratio(None|int|tuple): upsample ratio. Only support 4x upsample currently.
        """

        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_classes = num_classes
        assert loss_mask is not None
        self.loss_mask = build_loss(loss_mask)
        if loss_reg is not None:
            self.loss_reg = build_loss(loss_reg)
        else:
            self.loss_reg = None

        self.P4_conv = ConvModule(self.in_channels, self.conv_out_channels,
                                  kernel_size=3, stride=1, padding=1,
                                  conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg)
        self.P4_1x7_conv = ConvModule(self.conv_out_channels,
                                      self.conv_out_channels,
                                      kernel_size=(1, 7), stride=(1, 1),
                                      padding=(0, 3), conv_cfg=self.conv_cfg,
                                      norm_cfg=self.norm_cfg)
        self.channel4_1x7_conv = ConvModule(self.in_channels,
                                            self.conv_out_channels,
                                            kernel_size=(1, 7), stride=(1, 1),
                                            padding=(0, 3),
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg)
        self.rpn4 = ConvModule(self.conv_out_channels, self.conv_out_channels,
                               3, padding=1, conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg)

        self.conv_logits_seg = nn.Conv2d(self.conv_out_channels, self.num_classes, 1)
        self.conv_logits_reg = nn.Conv2d(self.conv_out_channels, 2, 1)

        self.relu = nn.ReLU(inplace=True)

        # Upsample the 4x feature map into original size
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods are '
                '"deconv", "nearest", "bilinear"'.format(upsample_method))
        self.upsample_method = upsample_method

        self.upsample_ratio = upsample_ratio
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                self.conv_out_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

    def init_weights(self):
        """ Weight initialization
        """
        for module in [self.conv_logits_seg, self.conv_logits_reg]:
            if module is not None:
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        if self.upsample_method == "deconv":
            nn.init.kaiming_normal_(
                self.upsample.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.upsample.bias, 0)

    @auto_fp16()
    def forward(self, x):
        """ Forward process. Only 4x feature maps is used to calculate loss.
            Additional feature maps supervising can be added if calculating resource is enough.

        Args:
            x (Tensor): input feature

        Returns:
            Tensor: (N x 1 x H x W ). predict results of aligned cell region mask, where N is batch size
            Tensor: (N x 2 x H x W ). predict results of global pyramid mask, where N is batch size

        """

        x_4 = x[0]
        x_p4 = self.P4_conv(x_4)
        x_4_1x7 = self.channel4_1x7_conv(x_4)
        x_p4_1x7 = self.P4_1x7_conv(x_p4)
        x_4 = x_p4_1x7 + x_p4 + x_4_1x7
        x_4 = self.rpn4(x_4)

        # predict results of aligned cell region mask and global pyramid mask
        mask_pred = self.conv_logits_seg(x_4)
        reg_pred = self.conv_logits_reg(x_4)

        # If upsample is defined, 4x feature maps will be upsampled to 1x feature maps for training
        if self.upsample is not None:
            assert self.upsample_ratio == 4, "Only support 4x upsample currently"
            mask_pred = self.upsample(mask_pred)

        return mask_pred, reg_pred

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, preds, mask_targets):
        """ Compute the loss.

        Args:
            preds (tuple): (mask_pred(Tensor), reg_pred(Tensor))
            mask_targets (tuple): (mask_target(Tensor),mask_weights(Tensor),geo_bond_target(Tensor),geo_weights(Tensor))

        Returns:
            dict[str, Tensor]: updated loss['loss_seg'] and loss['loss_reg'].
        """

        mask_pred, reg_pred = preds
        mask_target, mask_weights, geo_bond_target, geo_weights = mask_targets

        loss = dict()
        loss["loss_seg"] = self.loss_mask(mask_pred, mask_target, weight=mask_weights)
        if self.loss_reg is not None:
            loss_reg = self.loss_reg(reg_pred, geo_bond_target, weight=geo_weights)
            loss["loss_reg"] = 0.3 * loss_reg

        return loss

    def get_target(self, gt_semantic_seg):
        """ Generating gt_mask for training(GPMA branch)

        Args:
            gt_semantic_seg(np.ndarray): (N, 6, H, W], where N is batch size
                gt_semantic_seg:[:,0]: gt_cell_region
                gt_semantic_seg:[:,1]: cell_region_weight, 1 Care / 0 Not Care
                gt_semantic_seg:[:,2:4]: gt_global_pyramid
                gt_semantic_seg:[:,4:6]: global_pyramid_weight, 1 Care / 0 Not Care

        Returns:
            Tensor: (N x 1 x H x W ). aligned cell region mask.
            Tensor: (N x H x W ). weight mask of target aligned cell region.
            Tensor: (N x 2 x H x W ). global pyramid mask.
            Tensor: (N x 2 x H x W ). weight mask of target global pyramid mask.
        """

        # aligned cell region mask
        score_map_target = gt_semantic_seg[:, 0:1, :, :].float()
        score_map_weights = gt_semantic_seg[:, 1, :, :].float()

        # global pyramid mask
        geo_bond_target = gt_semantic_seg[:, 2:4, :, :]
        geo_bond_weights_target = gt_semantic_seg[:, 4:6, :, :]

        return score_map_target, score_map_weights, geo_bond_target, geo_bond_weights_target

    def get_seg_masks(self, preds, img_metas, pad_shape):
        """ Get the final predictions of GPMA branch in testing.

        Args:
            preds (tuple(Tensor)): (mask_pred, reg_pred)
            img_metas(dict): image meta infos
            pad_shape (tuple): image size (pad_shape)

        Returns:
            list(Tensor):  [prediction of cells region mask, global pyramid mask horizontal, lpma targets in vertical]
        """

        mask_pred, reg_pred = preds
        h_pad, w_pad = pad_shape
        cell_region_mask, gp_mask_hor, gp_mask_ver = [], [], []
        for i, meta in enumerate(img_metas):
            h_img, w_img, _ = meta['img_shape']
            h_ori, w_ori, _ = meta['ori_shape']
            if isinstance(mask_pred, torch.Tensor):
                mask_pred = mask_pred.sigmoid().cpu().numpy()
            if isinstance(reg_pred, torch.Tensor):
                reg_pred = reg_pred.cpu().numpy()

            mask_pred_ = mask_pred[i, 0, :, :]
            mask_pred_resize = mmcv.imresize(mask_pred_, (w_pad, h_pad))
            mask_pred_resize = mmcv.imresize(mask_pred_resize[:h_img, :w_img], (w_ori, h_ori))
            mask_pred_resize = (mask_pred_resize > 0.5)
            cell_region_mask.append(mask_pred_resize)

            reg_pred1_ = reg_pred[i, 0, :, :]
            reg_pred2_ = reg_pred[i, 1, :, :]
            reg_pred1_resize = mmcv.imresize(reg_pred1_, (w_pad, h_pad))
            reg_pred2_resize = mmcv.imresize(reg_pred2_, (w_pad, h_pad))
            reg_pred1_resize = mmcv.imresize(reg_pred1_resize[:h_img, :w_img], (w_ori, h_ori))
            reg_pred2_resize = mmcv.imresize(reg_pred2_resize[:h_img, :w_img], (w_ori, h_ori))
            gp_mask_hor.append(reg_pred1_resize)
            gp_mask_ver.append(reg_pred2_resize)

        return list(zip(cell_region_mask, gp_mask_hor, gp_mask_ver))
