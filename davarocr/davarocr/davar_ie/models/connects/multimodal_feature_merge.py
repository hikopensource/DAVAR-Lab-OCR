"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    feature_merge.py
# Abstract       :    merge multimodal features (vision and semantic).

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
import torch
from torch import nn
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models import CONNECTS
from davarocr.davar_common.utils import get_root_logger


@CONNECTS.register_module()
class MultiModalFusion(nn.Module):
    """Implementation of adaptive multi-modality context fusion in TRIE [1]

    Ref: [1] TRIE: End-to-End Text Reading and Information Extraction for Document Understanding. ACM MM-20.
                <https://arxiv.org/pdf/2005.13118.pdf>`_
    """
    def __init__(self,
                 visual_dim,
                 semantic_dim,
                 merge_type='Sum',
                 dropout_ratio=0.1,
                 with_extra_fc=True,
                 shortcut=False
                 ):
        """
        Args:
            visual_dim (list): the dim of visual features, e.g. [256]
            semantic_dim (list): the dim of semantic features, e.g. [256]
            merge_type (str): fusion type, e.g. 'Sum', 'Concat', 'Weighted'
            dropout_ratio (float): dropout ratio of fusion features
            with_extra_fc (bool): whether add extra fc layers for adaptation
            shortcut (bool): whether add shortcut connection
        """
        super().__init__()

        # merge param
        self.merge_type = merge_type
        self.visual_dim = visual_dim
        self.textual_dim = semantic_dim
        self.with_extra_fc = with_extra_fc
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)

        if self.merge_type == 'Sum':
            assert len(self.visual_dim) == len(self.textual_dim)
        elif self.merge_type == 'Concat':
            pass
        elif self.merge_type == 'Weighted':
            assert len(self.visual_dim) == len(self.textual_dim)
            self.total_num = len(self.visual_dim)

            # vis projection
            self.vis_proj = nn.ModuleList()
            self.vis_proj_relu = nn.ModuleList()

            # text projection
            self.text_proj = nn.ModuleList()
            self.text_proj_relu = nn.ModuleList()

            self.alpha_proj = nn.ModuleList()
            for idx in range(self.total_num):
                if self.with_extra_fc:
                    self.vis_proj.append(nn.Linear(self.visual_dim[idx], self.visual_dim[idx]))
                    self.text_proj.append(nn.Linear(self.textual_dim[idx], self.textual_dim[idx]))
                self.alpha_proj.append(nn.Linear(self.visual_dim[idx] + self.textual_dim[idx], self.visual_dim[idx]))

        else:
            raise "Unknown merge type {}".format(self.merge_type)

        self.dropout = nn.Dropout(dropout_ratio)

        # visual context
        self.visual_ap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Multimodal Feature Merge:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, visual_feat=None, textual_feat=None):
        """ Forward computation

        Args:
            visual_feat (list(Tensor)): visual feature maps, in shape of [L x C x H x W] x B
            textual_feat (Tensor): textual feature maps, in shape of B x L x C
        Returns:
            Tensor: fused feature maps, in shape of [B x L x C]
        """
        assert len(visual_feat) == textual_feat.size(0)
        visual_feat = torch.cat(visual_feat, 0)
        visual_feat = self.visual_ap(visual_feat).squeeze(2).squeeze(2)
        visual_feat = visual_feat.view(textual_feat.size(0), -1, visual_feat.size(-1))

        # feature merge
        if self.merge_type == 'Sum':
            x_sentence = visual_feat + textual_feat
        elif self.merge_type == 'Concat':
            x_sentence = torch.cat((visual_feat, textual_feat), -1)
        else:
            if self.with_extra_fc:
                visual_feat = self.relu(self.vis_proj[0](visual_feat))
                textual_feat = self.relu(self.text_proj[0](textual_feat))

            alpha = torch.sigmoid(self.alpha_proj[0](torch.cat((visual_feat, textual_feat), -1)))
            x_sentence = alpha * visual_feat + (1 - alpha) * textual_feat

        x_sentence = self.dropout(x_sentence)
        return x_sentence
