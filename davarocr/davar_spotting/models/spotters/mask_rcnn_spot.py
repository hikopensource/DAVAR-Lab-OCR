"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_spot.py
# Abstract       :    Mask-RCNN based text spotting model

# Current Version:    1.0.0
# Date           :    2021-09-01
####################################################################################################
"""
import torch
import numpy as np

from mmdet.core import BitmapMasks, bbox2roi

from .base import TwoStageEndToEnd
from ..builder import SPOTTER

@SPOTTER.register_module()
class MaskRCNNSpot(TwoStageEndToEnd):
    """ Mask rcnn for text spotting"""

    def __init__(self,
                 backbone,
                 rcg_roi_extractor,
                 rcg_sequence_head,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 rcg_backbone=None,
                 rcg_neck=None,
                 rcg_transformation=None,
                 rcg_sequence_module=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,):

        super().__init__(
            backbone=backbone,
            rcg_roi_extractor=rcg_roi_extractor,
            rcg_sequence_head=rcg_sequence_head,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            rcg_backbone=rcg_backbone,
            rcg_neck=rcg_neck,
            rcg_transformation=rcg_transformation,
            rcg_sequence_module=rcg_sequence_module,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def simple_test(self,
                    img,
                    img_metas,
                    gt_texts=None,
                    rescale=False,
                    proposals=None):
        """ Forward test process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_texts (list(list(str))): transcriptions for text recognition:
            rescale (boolean): if the image be re-scaled
            proposals (list(list(np.array))): proposals for detection:
            **kwargs: other parameters

        Returns:
            dict: formated inference results
        """

        # ===========================text detection branch================
        results = dict()

        # Feature extraction
        feats = self.extract_feat(img)

        # Get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals

        # Get text detection results (including bboxes, masks coresponding to each text instance)
        det_results = self.roi_head.simple_test(feats, proposal_list, img_metas)
        det_bboxes, det_masks = zip(*det_results)
        det_bboxes = [np.concatenate(box, axis=0) for box in det_bboxes]
        det_masks = [np.concatenate(mask, axis=0) for mask in det_masks]

        results['bboxes_preds'] = det_bboxes # list(np.array(N, 5)), len=B
        results['seg_preds'] = det_masks # list(np.array(N, H, W)), len=B

        if sum([box.shape[0] for box in det_bboxes]) == 0:
            # non positive gts in current batch
            results = [{"points" : [], "texts" : []}] * img.size(0)
            return results

        # ===========================text recognation branch================
        _bboxes = [torch.from_numpy(per_bbox[:, :4]).to(img.device) for per_bbox in det_bboxes]
        _masks = [BitmapMasks(mask.astype(np.uint8), mask.shape[1], mask.shape[2]) for mask in det_masks]
        recog_rois = bbox2roi(_bboxes)

        # Extract feature according to RoI and masks
        recog_feats = self.recog_roi_extractor(
            feats[:self.recog_roi_extractor.num_inputs], recog_rois, _masks)

        # Feature transformation
        if self.with_recog_transformation:
            recog_feats = self.recog_transformation(recog_feats)

        # Further extract feature for text recognation
        if self.with_recog_backbone:
            recog_feats = self.recog_backbone(recog_feats)

        # Extract multi-scale feature
        if self.with_recog_neck:
            recog_feats = self.recog_neck(recog_feats)

        # N x C x H x W -> N x C x 1 x HW
        batch, old_c, old_h, old_w = recog_feats.size()
        recog_feats = recog_feats.view(batch, old_c, 1, old_h * old_w)

        if self.keepdim_test:
            # N x C x 1 x HW -> N x 1 x HW x C
            if self.use_permute_test:
                recog_feats = recog_feats.permute(0, 2, 3, 1)
        else:
            # N x C x 1 x HW -> N x C x HW -> N x HW x C
            recog_feats = recog_feats.squeeze(2).permute(0, 2, 1)

        # Semantic feature extraction
        if self.with_recog_sequence_module:
            recog_contextual_feature = self.recog_sequence_module(recog_feats)
        else:
            recog_contextual_feature = recog_feats

        # Text recognition
        recog_target = self.recog_sequence_head.get_target(gt_texts)
        recog_prediction = self.recog_sequence_head(recog_contextual_feature.contiguous(), recog_target, is_train=False)
        text_preds = self.recog_sequence_head.get_pred_text(recog_prediction, self.test_cfg.batch_max_length)

        # Assign the text recognition result to the corresponding img
        det_texts = []
        recog_rois = recog_rois.cpu().numpy()
        text_preds = np.array(text_preds)
        for batch_id in range(img.size(0)):
            ind = np.where(recog_rois == batch_id)[0]
            text = text_preds[ind].tolist()
            det_texts.append(text)

        results['text_preds'] = det_texts  # list(B, N)

        # Post-processing
        if self.post_processor is not None:
            results = self.post_processor.post_processing(results, img_metas)

        return results

    def aug_test(self, imgs, img_metas, rescale=False):
        """ Forward aug_test. Not implemented."""
        raise NotImplementedError
