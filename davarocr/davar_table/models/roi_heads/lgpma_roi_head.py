"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    lgpma_roi_head.py
# Abstract       :    The main pipeline definition of lgpma_roi_head

# Current Version:    1.0.1
# Date           :    2022-03-09
# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

import torch
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.builder import HEADS
from mmdet.core import bbox2roi, bbox2result


@HEADS.register_module()
class LGPMARoIHead(StandardRoIHead):
    """ RoI head used in LGPMA, which including bbox head and lpma head [1].

    Ref: Qiao L, Li Z, Cheng Z, et al. LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask
    Alignment[J]. arXiv preprint arXiv:2105.06224, 2021. (Accepted by ICDAR 2021, Best Industry Paper)

    """

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on the values
                of these keys see `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with shape (num_gts, 4) in
                [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box used if the architecture supports a
                segmentation task.

        Returns:
            dict[str: Tensor]: a dictionary of loss components
        """

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()

        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas, gt_bboxes)
            losses.update(mask_results['loss_mask'])

        return losses

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas, gt_bboxes=None):

        """Run forward function and calculate loss for mask head with lpma in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            bbox_feats (Tensor): bbox_feats
            gt_masks (None | Tensor) : true segmentation masks for each box used if the architecture supports a
                segmentation task.
            img_metas (list[dict]): list of image info dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on the values
                of these keys see `mmdet/datasets/pipelines/formatting.py:Collect`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg, gt_bboxes)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):

        """ Forward test process

        Args:
            x(Tensor): input feature
            proposal_list (list[Tensor]): Proposals of each image.
            img_metas(dict): image meta-info
            proposals(None | list): if region proposals is assigned before, using it
            rescale(boolean): if the image be re-scaled

        Returns:
            list[Tensor]: list consists of bbox_results, segm_results and segm_results_soft
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        assert self.with_mask, 'Mask head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            segm_results, segm_results_soft = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results, segm_results_soft

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        segm_results, segm_results_soft = self.simple_test_mask(
            x, img_metas, det_bboxes, det_labels, rescale=rescale)

        return list(zip(bbox_results, segm_results, segm_results_soft))

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """ Simple test for lp_mask head without augmentation

        Args:
            x(Tensor): input feature

            img_metas(dict): image meta-info

            det_bboxes(list[Tensor]): bboxes of aligned cells

            det_labels(list[Tensor]): labels of aligned cells

            rescale(boolean): if the image be re-scaled

        Returns:
            segm_results(list[Tensor]): masks resuls of text regions
            segm_results_soft(list[Tensor]): masks resuls of local pyramid masks
        """
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        num_imgs = len(det_bboxes)

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
            segm_results_soft = [[[] for _ in range(2)]
                                 for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            if torch.onnx.is_in_onnx_export():
                # avoid mask_pred.split with static number of prediction
                mask_preds = []
                mask_pred_softs = []
                _bboxes = []
                for i, boxes in enumerate(det_bboxes):
                    boxes = boxes[:, :4]
                    if rescale:
                        boxes *= scale_factors[i]
                    _bboxes.append(boxes)
                    img_inds = boxes[:, :1].clone() * 0 + i
                    mask_rois = torch.cat([img_inds, boxes], dim=-1)
                    mask_result = self._mask_forward(x, mask_rois)
                    mask_preds.append(mask_result['mask_pred'][:, 0:-2, :, :])
                    mask_pred_softs.append(mask_result['mask_pred'][:, -2:, :, :])
            else:
                _bboxes = [
                    det_bboxes[i][:, :4] *
                    scale_factors[i] if rescale else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                mask_results = self._mask_forward(x, mask_rois)
                mask_pred = mask_results['mask_pred']
                # split mask pred and lpma pred
                mask_pred, mask_pred_soft = mask_pred[:, 0:-2, :, :], mask_pred[:, -2:, :, :]
                # split batch mask prediction back to each image
                num_mask_roi_per_img = [
                    det_bbox.shape[0] for det_bbox in det_bboxes
                ]
                mask_preds = mask_pred.split(num_mask_roi_per_img, 0)
                mask_pred_softs = mask_pred_soft.split(num_mask_roi_per_img, 0)

            # apply soft mask post-processing to each image individually
            segm_results, segm_results_soft = [], []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                    segm_results_soft.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_result_soft = self.mask_head.get_seg_lpmasks(
                        mask_pred_softs[i], _bboxes[i], det_labels[i],
                        ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
                    segm_results_soft.append(segm_result_soft)

        return segm_results, segm_results_soft
