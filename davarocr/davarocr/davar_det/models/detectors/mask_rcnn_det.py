"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mask_rcnn_det.py
# Abstract       :    Mask-RCNN based text detection model

# Current Version:    1.0.0
# Date           :    2020-05-31
####################################################################################################
"""
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
from davarocr.davar_common.core import build_postprocess


@DETECTORS.register_module()
class MaskRCNNDet(TwoStageDetector):
    """ Mask rcnn for text detection"""
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """ Integrate with postprocessing(get the contour of mask region) for mask-rcnn model.

        Args:
            img (Tensor): Input image
            img_meta:(dict): images meta information
            proposals: (list(Tensor)): proposal list
            rescale(bool): whether the image be rescaled

        Returns:

        """
        results = super().simple_test(img=img, img_metas=img_metas, proposals=proposals, rescale=rescale)

        if hasattr(self.test_cfg, 'postprocess'):
            post_processor = build_postprocess(self.test_cfg.postprocess)
            results = post_processor(results)
        return results

    def aug_test(self, imgs, img_metas, rescale=False, **kwargs):
        """ Integrate with postprocessing(get the contour of mask region) for mask-rcnn model.

          Args:
              img (list(Tensor)): Input images
              img_meta:( list(dict)): images meta information
              rescale(bool): whether the image be rescaled

          Returns:

          """
        results = super().aug_test(imgs=imgs, img_metas=img_metas, rescale=rescale)

        if hasattr(self.test_cfg, 'postprocess'):  # 添加后处理
            post_processor = build_postprocess(self.test_cfg.postprocess)
            results = post_processor(results)

        return results
