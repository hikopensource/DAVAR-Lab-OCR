"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    spatial_temporal_east_det.py
# Abstract       :    YORO detection model

# Current Version:    1.0.0
# Date           :    2021-05-25
####################################################################################################
"""
from mmdet.models.builder import DETECTORS
from mmdet.models import builder

from davarocr.davar_det.models.detectors.seg_based_det import SegBasedDet
from davarocr.davar_common.core import build_postprocess


@DETECTORS.register_module()
class SpatialTempoEASTDet(SegBasedDet):
    """ YORO detection model use EAST as backbone

    """
    def __init__(self,
                 backbone,
                 neck=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """
      Args:
            backbone(dict): network backbone (e.g. ResNet)
            neck(dict): network neck (e.g., EASTMerge)
            head(dict): head for loss calculation (e.g., EASTHead)
            train_cfg(dict): related parameters for training
            test_cfg(dict): related parameters for test
            pretrained(dict): pretrained model
        """
        super().__init__(backbone=backbone, neck=neck, mask_head=mask_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, pretrained=pretrained)

        # Config set
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.head_cfg = {'head_cfg': self.train_cfg}

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        # You can set the train cfg to decide whether to fix backbone
        if self.train_cfg and self.train_cfg.get('fix_backbone', False):
            for param in self.parameters():
                param.requires_grad = False

        mask_head = dict(mask_head, **self.head_cfg)

        self.mask_head = builder.build_head(mask_head)

        self.init_weights(pretrained=pretrained)

        if hasattr(self.test_cfg, 'postprocess'):
            self.post_processor = build_postprocess(self.test_cfg.postprocess)
        else:
            self.post_processor = None

    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      **kwargs
                      ):
        """Forward training process and loss computing

        Args:
            img (list[Tensor]): input images
            img_metas(dict) : image meta-info
            gt_masks(np.ndarray): ground-truth label for training

        Returns:
            dict: losses for training data
        """
        losses = dict()

        # You can set the train cfg to decide whether to fix backbone
        if self.train_cfg and self.train_cfg.get('fix_backbone', False):
            self.backbone.eval()
            self.neck.eval()

        x = self.extract_feat(img)

        mask_pred = self.mask_head(x, img_metas)

        # Get ground-truth label
        mask_targets = self.mask_head.get_target(gt_masks)

        # Compute loss
        loss_mask = self.mask_head.loss(mask_pred, mask_targets)

        # Update loss
        losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_meta, **kwargs):
        """Forward inference

        Args:
            img(Tensor): input image
            img_meta(dict): image meta-info

        Returns:
            dict: predicted results.  e.g. {'bboxes': [{'points':[[x1, y1, ...., xn, yn],[]...]}, {},....],}.
        """
        final_result = dict()

        x = self.extract_feat(img)

        # If the previous features exist, we only forward last frame, and save this frame's info
        if img_meta[-1]['pre_features'] is not None:
            final_result['pre_features'] = x
            final_result['img_metas'] = img_meta

        # Or we forward window size frames and only to save [1:] previous features for next forward
        else:
            final_result['pre_features'] = x[1:]
            final_result['img_metas'] = img_meta[1:]

        results = self.mask_head.simple_test(x, img_meta)

        if self.post_processor is not None:
            results = self.post_processor.post_processing(results, img_meta)

        final_result['bboxes'] = results

        return final_result

    def aug_test(self, img, img_meta):
        raise NotImplementedError
