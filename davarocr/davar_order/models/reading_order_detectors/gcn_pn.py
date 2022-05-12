"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    gcn_pn.py
# Abstract       :    Graph Convolution Networks with Pointer-Net

# Current Version:    1.0.0
# Author         :    Can Li
# Date           :    2022-05-11
##################################################################################################
"""
import torch
from mmdet.core import bbox2roi
import copy

from mmdet.models import build_head
from davarocr.davar_spotting.models import SPOTTER
from davarocr.davar_spotting.models import TwoStageEndToEnd
from mmdet.models import build_roi_extractor
from davarocr.davar_common.models.builder import build_connect

@SPOTTER.register_module()
class GCN_PN(TwoStageEndToEnd):
    """Implementation of encoder in GCN-PN

    Ref: An End-to-End OCR Text Re-organization Sequence Learning for Rich-text Detail Image Comprehension. ECCV-20.
    """
    def __init__(self,
                 backbone,
                 rcg_backbone=None,
                 rcg_roi_extractor=None,
                 rcg_sequence_head=None,
                 infor_context_module=None,
                 infor_node_cls_head=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 rcg_neck=None,
                 rcg_transformation=None,
                 rcg_sequence_module=None,
                 infor_roi_extractor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ):
        """
        Args:
            backbone(dict): backbone of detection model (e.g. ResNet)
            rcg_backbone(dict): backbone of rcg model (e.g. ResNet32)
            rcg_roi_extractor(dict): roi_extractor for rcg
            rcg_sequence_head(dict): sequence head in rcg (e.g. AttentionHead)
            infor_context_module(dict): context module for IE (e.g. MultiModalContextModule)
            infor_node_cls_head(dict): node cls head for IE (e.g. ClsHead)
            neck(dict): necks of the detection model (e.g., FPN)
            rpn_head(dict): rpn head
            roi_head(dict): roi head
            rcg_neck(dict): necks of rcg model (default None)
            rcg_transformation(dict): transformation of rcg model
            rcg_sequence_module(dict): sequence module of rcg model (e.g. CascadeRNN)
            train_cfg(dict): default None
            test_cfg(dict): definition of postprocess/ prune_model
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        super().__init__(
            backbone=backbone,
            rcg_backbone=rcg_backbone,
            rcg_roi_extractor=rcg_roi_extractor,
            rcg_sequence_head=rcg_sequence_head,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            rcg_neck=rcg_neck,
            rcg_transformation=rcg_transformation,
            rcg_sequence_module=rcg_sequence_module,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        self.infor_context_module = build_connect(infor_context_module)
        if infor_roi_extractor is not None:
            self.infor_roi_extractor = build_roi_extractor(infor_roi_extractor)

        if infor_node_cls_head is not None:
            self.infor_node_cls_head = build_head(infor_node_cls_head)


    @property
    def with_infor_roi_extractor(self):
        return hasattr(self, 'infor_roi_extractor') and self.infor_roi_extractor is not None

    def with_infor_context_module(self):
        """

        Returns:
            Determine the model with the infor_context_module or not
        """
        return hasattr(self, 'infor_context_module') and self.infor_context_module is not None

    @property
    def with_infor_node_cls_head(self):
        """

        Returns:
            Determine the model with the infor_node_cls_head or not
        """
        return hasattr(self, 'infor_node_cls_head') and self.infor_node_cls_head is not None


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """ Forward train process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            gt_bboxes (list(Tensor): Tensor bboxes for each image, in [x_tl, y_tl, x_br, y_br] order.
            gt_labels (list(Tensor): category labels for each bbox
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
        Returns:
            dict: all losses in a dict
        """
        losses = dict()

        # remap labels
        info_labels = copy.deepcopy(gt_labels)
        for idx in range(len(gt_labels)):
            info_labels[idx][gt_labels[idx]>0] = 1

        ## ===================== text detection branch ====================
        x = self.extract_feat(img)
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, gt_bboxes,
                                                                    gt_labels=None,
                                                                    gt_bboxes_ignore=gt_bboxes_ignore,
                                                                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # ROI forward and loss
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, info_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        if sum([box.shape[0] for box in gt_bboxes]) == 0:
            # non positive gts in current batch, we just not train the recognition branch
            return losses

        recog_rois = bbox2roi(gt_bboxes)

        info_feat_list = []
        if self.with_infor_roi_extractor:
            infor_feats = self.infor_roi_extractor(x[:self.infor_roi_extractor.num_inputs], recog_rois)
            info_feat_list.append(infor_feats)

        #get relation between bboxes
        batch_bboxes_relation = []
        bboxes_num = []
        for batch in range(len(gt_bboxes)):
            gt_bbox = gt_bboxes[batch]
            bboxes_num.append(gt_bbox.shape[0])
            x = gt_bbox[:,0]
            y = gt_bbox[:,1]
            l = gt_bbox[:,2]-gt_bbox[:,0]+1e-5
            h = gt_bbox[:,3]-gt_bbox[:,1]+1e-5
            gt_bbox_num = gt_bbox.shape[0]
            r = []
            r.append(torch.abs(x.unsqueeze(0)-x.unsqueeze(1)).unsqueeze(-1))  #delta x
            r.append(torch.abs(y.unsqueeze(0)-y.unsqueeze(1)).unsqueeze(-1))  #delta y
            r.append(((l/h)[:,None].repeat(1,gt_bbox_num)).unsqueeze(-1))     #l_i/h_i
            r.append((l.unsqueeze(0)/h.unsqueeze(1)).unsqueeze(-1))           #l_j/h_i
            r.append((h.unsqueeze(0)/h.unsqueeze(1)).unsqueeze(-1))           #h_j/h_i
            r.append((h.unsqueeze(0)/l.unsqueeze(1)).unsqueeze(-1))           #h_j/l_i
            r.append((l.unsqueeze(0)/l.unsqueeze(1)).unsqueeze(-1))           #l_j/l_i
            r = torch.cat(r,dim=-1)
            batch_bboxes_relation.append(r)

        all_bboxes_num = infor_feats.size(0)
        infor_feats = infor_feats.view(all_bboxes_num,-1)
        batch_node_feats = torch.split(infor_feats, bboxes_num)
        
        # update node and edge features through GCN
        batch_node_embedding, batch_edge_embedding = self.infor_context_module(batch_bboxes_relation, batch_node_feats)

        # pointer-net decoder
        batch_z_g = []
        batch_z_l = []

        for i in range(len(batch_node_embedding)):
            node_embedding = batch_node_embedding[i]
            edge_embedding = batch_edge_embedding[i]
            hidden_size = edge_embedding.size(-1)
            Z_G = torch.mean(node_embedding,dim=0)
            Z_L = torch.mean(edge_embedding.view(-1,hidden_size),dim=0)
            batch_z_g.append(Z_G)
            batch_z_l.append(Z_L)

        loss = self.infor_node_cls_head.loss(batch_z_g, batch_z_l, batch_node_embedding, gt_labels)
        losses['node_cls_loss'] = loss
        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    rescale=False,
                    proposals=None,
                    **kwargs):
        """ Forward test process

        Args:
            img(Tensor): input images
            img_metas(dict): image meta infos
            gt_bboxes (list(Tensor): Tensor bboxes for each image, in [x_tl, y_tl, x_br, y_br] order.
            rescale(boolean): if the image be re-scaled
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            list: in shape of [N], decoding labels of pred.
        """
        ## ===========================text detection branch================
        x = self.extract_feat(img)
        gt_bboxes = gt_bboxes[0]
        recog_rois = bbox2roi(gt_bboxes)

        info_feat_list = []
        if self.with_infor_roi_extractor:
            infor_feats = self.infor_roi_extractor(x[:self.infor_roi_extractor.num_inputs], recog_rois)
            info_feat_list.append(infor_feats)

        #get relation between bboxes
        batch_bboxes_relation = []
        bboxes_num = []
        for batch in range(len(gt_bboxes)):
            gt_bbox = gt_bboxes[batch]
            bboxes_num.append(gt_bbox.shape[0])
            x = gt_bbox[:,0]
            y = gt_bbox[:,1]
            l = gt_bbox[:,2]-gt_bbox[:,0]+1e-5
            h = gt_bbox[:,3]-gt_bbox[:,1]+1e-5
            gt_bbox_num = gt_bbox.shape[0]
            r = []
            r.append(torch.abs(x.unsqueeze(0)-x.unsqueeze(1)).unsqueeze(-1))#delta x
            r.append(torch.abs(y.unsqueeze(0)-y.unsqueeze(1)).unsqueeze(-1))#delta y
            r.append(((l/h)[:,None].repeat(1,gt_bbox_num)).unsqueeze(-1))#l_i/h_i
            r.append((l.unsqueeze(0)/h.unsqueeze(1)).unsqueeze(-1))#l_j/h_i
            r.append((h.unsqueeze(0)/h.unsqueeze(1)).unsqueeze(-1))#h_j/h_i
            r.append((h.unsqueeze(0)/l.unsqueeze(1)).unsqueeze(-1))#h_j/l_i
            r.append((l.unsqueeze(0)/l.unsqueeze(1)).unsqueeze(-1))#l_j/l_i
            r = torch.cat(r,dim=-1)
            batch_bboxes_relation.append(r)

        all_bboxes_num = infor_feats.size(0)
        infor_feats = infor_feats.view(all_bboxes_num,-1)
        batch_node_feats = torch.split(infor_feats, bboxes_num)

        # update node and edge features through GCN
        batch_node_embedding, batch_edge_embedding = self.infor_context_module(batch_bboxes_relation, batch_node_feats)

        # pointer-net decoder
        batch_z_g = []
        batch_z_l = []

        for i in range(len(batch_node_embedding)):
            node_embedding = batch_node_embedding[i]
            edge_embedding = batch_edge_embedding[i]
            hidden_size = edge_embedding.size(-1)
            Z_G = torch.mean(node_embedding,dim=0)
            Z_L = torch.mean(edge_embedding.view(-1,hidden_size),dim=0)
            batch_z_g.append(Z_G)
            batch_z_l.append(Z_L)

        results = self.infor_node_cls_head.get_predict(batch_z_g, batch_z_l, batch_node_embedding)

        return results
