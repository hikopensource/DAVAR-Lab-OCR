"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mm_layout_feature_merge.py
# Abstract       :    merge multimodal features in VSR.

# Current Version:    1.0.0
# Date           :    2021-12-06
##################################################################################################
"""
import torch
from davarocr.davar_common.models import CONNECTS
from davarocr.davar_ie.models import MultiModalFusion


@CONNECTS.register_module()
class VSRFeatureMerge(MultiModalFusion):
    """Multimodal feature fusion used in VSR."""
    def __init__(self,
                 visual_dim,
                 semantic_dim,
                 merge_type='Sum',
                 dropout_ratio=0.1,
                 with_extra_fc=True,
                 shortcut=False
                 ):
        """Multimodal feature merge used in VSR.

        Args:
            visual_dim (list): the dim of visual features, e.g. [256]
            semantic_dim (list): the dim of semantic features, e.g. [256]
            merge_type (str): fusion type, e.g. 'Sum', 'Concat', 'Weighted'
            dropout_ratio (float): dropout ratio of fusion features
            with_extra_fc (bool): whether add extra fc layers for adaptation
            shortcut (bool): whether add shortcut connection
        """
        super().__init__(
            visual_dim=visual_dim,
            semantic_dim=semantic_dim,
            merge_type=merge_type,
            dropout_ratio=dropout_ratio,
            with_extra_fc=with_extra_fc,
            shortcut=shortcut
        )

    def forward(self, visual_feat=None, textual_feat=None):
        """ Forward computation

        Args:
            visual_feat (list(Tensor)): visual feature maps, in shape of [L x C x H x W] x B
            textual_feat (Tensor): textual feature maps, in shape of B x L x C
        Returns:
            Tensor: fused feature maps, in shape of [B x L x C]
        """
        assert len(visual_feat) == len(textual_feat)

        # feature merge
        if self.merge_type == 'Sum':
            merged_feat = [per[0]+per[1] for per in zip(visual_feat, textual_feat)]
        elif self.merge_type == 'Concat':
            merged_feat = [torch.cat(per, 1) for per in zip(visual_feat, textual_feat)]
        else:
            assert self.total_num == len(visual_feat) or self.total_num == 1
            merged_feat = []

            idx = 0
            for per_vis, per_text in zip(visual_feat, textual_feat):
                per_vis = per_vis.permute(0, 2, 3, 1)
                per_text = per_text.permute(0, 2, 3, 1)
                if self.with_extra_fc:
                    per_vis = self.relu(self.vis_proj[idx](per_vis))
                    per_text = self.relu(self.text_proj[idx](per_text))

                alpha = torch.sigmoid(self.alpha_proj[idx](torch.cat((per_vis, per_text), -1)))
                if self.shortcut:
                    # shortcut
                    x_sentence = per_vis + alpha * per_text
                else:
                    # selection
                    x_sentence = alpha * per_vis + (1 - alpha) * per_text

                x_sentence = x_sentence.permute(0,3,1,2).contiguous()
                merged_feat.append(x_sentence)
                if self.total_num == len(visual_feat):
                    idx += 1

        return tuple(merged_feat)
