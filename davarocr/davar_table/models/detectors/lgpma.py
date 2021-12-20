"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    lgpma.py
# Abstract       :    The main pipeline definition of LGPMA model

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

from torch import nn
from mmdet.models import builder
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
from davarocr.davar_common.core import build_postprocess


@DETECTORS.register_module()
class LGPMA(TwoStageDetector):
    """Implementation of LGPMA detector model.

    Ref: Qiao L, Li Z, Cheng Z, et al. LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask
     Alignment[J]. arXiv preprint arXiv:2105.06224, 2021. (Accepted by ICDAR 2021, Best Industry Paper)

    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 global_seg_head=None
                 ):
        """
        Args:
            backbone(dict): network backbone (e.g. ResNet)
            rpn_head(dict): rpn head
            roi_head(dict): roi_head
            train_cfg(dict): related parameters for training
            test_cfg(dict): related parameters for test
            neck(dict): network neck (e.g., FPN)
            pretrained(dict): pretrained model
            global_seg_head: global segmentation head
        """

        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if global_seg_head is not None:
            self.global_seg_head = builder.build_head(global_seg_head)
            if isinstance(self.global_seg_head, nn.Sequential):
                for module in self.global_seg_head:
                    module.init_weights()
            else:
                self.global_seg_head.init_weights()
        else:
            self.global_seg_head = None

    @property
    def with_global_seg(self):
        """
        Returns:
            boolean:Determine the model with the global_seg_head or not
        """

        return hasattr(self, 'global_seg_head') and self.global_seg_head is not None

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_semantic_seg=None,
                      **kwargs):
        """ Forward train process.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box used if the architecture supports a
                segmentation task.
            proposals: override rpn proposals with custom proposals. Use when `with_rpn` is False.
            gt_semantic_seg (None | Tensor) : true global segmentation masks for the whole image used if the
                architecture supports a global segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        # global forward and loss
        if self.with_global_seg:
            # Change shape to compatible DavarDefaultFormatBundle
            gt_semantic_seg = gt_semantic_seg[:, 0, :, :, :]
            seg_pred = self.global_seg_head(x)
            seg_targets = self.global_seg_head.get_target(gt_semantic_seg)
            loss_global_seg = self.global_seg_head.loss(seg_pred, seg_targets)
            losses.update(loss_global_seg)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """ Forward test process

        Args:
            img(Tensor): input images
            img_metas(dict): image meta infos
            proposals(None | list): if region proposals is assigned before, using it
            rescale(boolean): if the image be re-scaled

        Returns:
            list(str): Format results, like [html of table1 (str), html of table2 (str), ...]

        """
        assert self.with_mask, 'Mask head must be implemented.'
        results = super(LGPMA, self).simple_test(img, img_metas, proposals, rescale)

        if self.with_global_seg:
            x = self.extract_feat(img)
            h_img, w_img = img.shape[1], img.shape[2]
            global_seg_pred = self.global_seg_head(x)
            global_seg_results = self.global_seg_head.get_seg_masks(global_seg_pred, img_metas, (h_img, w_img))
            results = [res + (seg,) for res, seg in zip(results, global_seg_results)]

        if hasattr(self.test_cfg, 'postprocess'):
            post_processor = build_postprocess(self.test_cfg.postprocess)
            results = post_processor(results)

        return results
