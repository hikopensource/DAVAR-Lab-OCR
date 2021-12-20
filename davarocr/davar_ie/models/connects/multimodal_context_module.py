"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    multimodal_context_module.py
# Abstract       :    compute multimodal context for each bbox/ node.

# Current Version:    1.0.1
# Date           :    2021-07-28
######################################################################################################
"""
import copy

import torch
from torch import nn
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models import CONNECTS
from davarocr.davar_common.models.builder import build_embedding
from davarocr.davar_common.models.builder import build_connect
from davarocr.davar_common.utils import get_root_logger

from .util import BertConfig


@CONNECTS.register_module()
class MultiModalContextModule(nn.Module):
    """Implementation of multimodal context computation in TRIE[1]

    Ref: [1] TRIE: End-to-End Text Reading and Information Extraction for Document Understanding. ACM MM-20.
                <https://arxiv.org/pdf/2005.13118.pdf>`_
    """
    def __init__(self,
                 textual_embedding,
                 multimodal_fusion_module,
                 textual_relation_module=None
                 ):
        """
        Args:
            textual_embedding (dict): config of textual/ node embedding, e.g. NodeEmbedding
            multimodal_fusion_module (dict): config of multimodal fusion module
            textual_relation_module (dict): relation module for textual context, e.g. BertEncoder
        """
        super().__init__()

        self.textual_embedding = build_embedding(textual_embedding)
        self.multimodal_fusion_module = build_connect(multimodal_fusion_module)

        if textual_relation_module is not None:
            raw_ber_config = textual_relation_module.get('config', None)
            assert raw_ber_config is not None
            bert_config = BertConfig(
                hidden_size=raw_ber_config.get('hidden_size', 768),
                num_hidden_layers=raw_ber_config.get('num_hidden_layers', 12),
                num_attention_heads=raw_ber_config.get('num_attention_heads', 12),
                intermediate_size=raw_ber_config.get('intermediate_size', 3072),
                hidden_act=raw_ber_config.get('hidden_act', "gelu"),
                hidden_dropout_prob=raw_ber_config.get('hidden_dropout_prob', 0.1),
                attention_probs_dropout_prob=raw_ber_config.get('attention_probs_dropout_prob', 0.1),
                layer_norm_eps=raw_ber_config.get('layer_norm_eps', 1e-12),
                output_attentions=raw_ber_config.get('output_attentions', False),
                output_hidden_states=raw_ber_config.get('output_hidden_states', False),
                is_decoder=raw_ber_config.get('is_decoder', False),
            )
            self.infor_bert_config = bert_config
            textual_relation_module['config'] = bert_config
            self.textual_relation_module = build_connect(textual_relation_module)

    @property
    def with_textual_relation_module(self):
        """
        Returns:
            Determine the model with the textual_relation_module or not
        """
        return hasattr(self, 'textual_relation_module') and self.textual_relation_module is not None

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Multimodal Context Module:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def pack_batch(self,
                   feat_all,
                   pos_feat,
                   img_meta,
                   info_labels,
                   bieo_labels=None):
        """ Forward computation

        Args:
            feat_all (list(Tensor)):
                0: visual features, in shape of N x C x H x W, where N is all text bboxes in a batch
                1: textual features, in shape of N x L x C, where L is max length of per text bbox
            pos_feat (list(Tensor)): box Tensor for each sample in a batch, e.g. B x 4
            img_meta (list(dict)): img_metas for each sample in a batch
            info_labels (list(Tensor)): category labels for all text bboxes
            bieo_labels (list(list(Tensor))): category labels for each characters in all bboxes if exist

        Returns:
            list(list(Tensor)):
                0: visual features for each sample, padded to same length, in Max_N x C x H x W
                1: textual features for each sample, padded to same length, in Max_N x L x C
        Returns:
            list(Tensor): box Tensor to same length, in Max_N x 4
        Returns:
            list(Tensor): category label Tensor to same length, in Max_N
        Returns:
            list(Tensor): bieo label Tensor to same length, in Max_N x L
        """
        img_feat_all = feat_all
        max_length = max([per_b.size(0) for per_b in pos_feat])

        # 0: visual feat, 1: textual feat
        batched_img_feat = [[], []]
        batched_pos_feat = []
        batched_img_label = []
        batched_img_bieo_label = []
        last_idx = 0

        # pack
        for i, _ in enumerate(pos_feat):
            # visual feat
            b_s = pos_feat[i].size(0)
            img_feat = img_feat_all[0]
            img_feat_size = list(img_feat.size())
            img_feat_size[0] = max_length - b_s
            batched_img_feat[0].append(
                torch.cat((img_feat[last_idx: last_idx + b_s], img_feat.new_full(img_feat_size, 0)), 0))

            # textual feat
            img_feat = img_feat_all[1]
            img_feat_size = list(img_feat.size())
            img_feat_size[0] = max_length - b_s
            batched_img_feat[1].append(
                torch.cat((img_feat[last_idx: last_idx + b_s], img_feat.new_full(img_feat_size, 0)), 0))

            # pos feat
            per_pos_feat = pos_feat[i]
            image_shape_h, image_shape_w = img_meta[i]['img_shape'][:2]
            per_pos_feat_expand = per_pos_feat.new_full((per_pos_feat.size(0), 4), 0)
            per_pos_feat_expand[:, 0] = per_pos_feat[:, 0]
            per_pos_feat_expand[:, 1] = per_pos_feat[:, 1]
            per_pos_feat_expand[:, 2] = per_pos_feat[:, 2]
            per_pos_feat_expand[:, 3] = per_pos_feat[:, 3]
            img_feat_size = list(per_pos_feat_expand.size())
            img_feat_size[0] = max_length - b_s

            # normalize
            per_pos_feat_expand[:, ::2] = per_pos_feat_expand[:, ::2] / image_shape_w
            per_pos_feat_expand[:, 1::2] = per_pos_feat_expand[:, 1::2] / image_shape_h

            batched_pos_feat.append(
                torch.cat((per_pos_feat_expand, per_pos_feat_expand.new_full(img_feat_size, 0)), 0))

            # classification labels
            if info_labels is not None:
                per_label = info_labels[i]
                img_feat_size = list(per_label.size())
                img_feat_size[0] = max_length - b_s
                batched_img_label.append(
                    torch.cat((per_label, per_label.new_full(img_feat_size, 255)), 0))

            # bieo labels
            if bieo_labels is not None:
                per_label = copy.deepcopy(bieo_labels[i])
                per_label = torch.tensor(per_label, dtype=torch.long).to(img_feat.device)

                img_feat_size = list(per_label.size())
                img_feat_size[0] = max_length - b_s
                batched_img_bieo_label.append(
                    torch.cat((per_label, per_label.new_full(img_feat_size, 255)), 0))

            last_idx += b_s
        return batched_img_feat, batched_pos_feat, batched_img_label, batched_img_bieo_label

    def forward(self,
                info_feat_list,
                pos_feat,
                img_metas,
                info_labels=None,
                bieo_labels=None):
        """ Forward computation

        Args:
            info_feat_list (list(Tensor)):
                0: visual features, in shape of N x C x H x W, where N is all text bboxes in a batch
                1: textual features, in shape of N x L x C, where L is max length of per text bbox
            pos_feat (list(Tensor)): box Tensor for each sample in a batch, e.g. B x 4
            img_metas (list(dict)): img_metas for each sample in a batch
            info_labels (list(Tensor)): category labels for all text bboxes
            bieo_labels (list(list(Tensor))): category labels for each characters in all bboxes if exist

        Returns:
            Tensor: fused feature maps, in shape of [B x Max_N x C]
        Returns:
            list(Tensor): category labels for bboxes in each sample
        Returns:
            list(Tensor): bieo label Tensor to same length, in B x Max_N x L
        """
        # pack data in a batch to same length
        batched_img_feat, batched_pos_feat, batched_img_label, batched_img_bieo_label = self.pack_batch(info_feat_list,
                                                                                                pos_feat=pos_feat,
                                                                                                img_meta=img_metas,
                                                                                                info_labels=info_labels,
                                                                                                bieo_labels=bieo_labels)
        origin_visual_context = batched_img_feat[0]
        origin_textual_context = batched_img_feat[1]
        # textual embedding
        textual_context = self.textual_embedding(origin_textual_context, batched_pos_feat)

        # textual context embedding
        if self.with_textual_relation_module:
            head_mask = [None] * self.infor_bert_config.num_hidden_layers

            tmp_batched_pos_bboxes = torch.stack(batched_pos_feat, 0)
            sentence_mask = (torch.sum(torch.abs(tmp_batched_pos_bboxes), dim=-1) != 0).float()
            sentence_mask = sentence_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            sentence_mask = (1.0 - sentence_mask) * -10000.0
            all_bert_outputs = self.textual_relation_module(textual_context,
                                                            attention_mask=sentence_mask[:, None, None, :],
                                                            head_mask=head_mask)
            textual_context = all_bert_outputs[0]

        # multimodal context fusion
        multimodal_contenxt = self.multimodal_fusion_module(origin_visual_context, textual_context)

        return multimodal_contenxt, batched_img_label, batched_img_bieo_label
