"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    multimodal_context_module_plusplus.py
# Abstract       :    compute multimodal context for each bbox/ node.

# Current Version:    1.0.1
# Date           :    2022-11-22
######################################################################################################
"""
import torch
import copy
import torch.nn as nn
from davarocr.davar_common.models import CONNECTS
from davarocr.davar_common.models.builder import build_embedding
from davarocr.davar_common.models.builder import build_connect
from .util import BertConfig


@CONNECTS.register_module()
class MultiModalContextModulePlusPlus(nn.Module):
    """Implementation of multimodal context computation in TRIE++[1]

    Ref: [1] TRIE++: Towards End-to-End Information Extraction from Visually Rich Documents
    """

    def __init__(self,
                 textual_embedding,
                 pos_embedding=None,
                 relative_pos_embedding=None,
                 textual_relation_module=None,
                 mode=0,
                 with_visual=True):
        """
        Args:
            textual_embedding (dict): config of textual/ node embedding, e.g. NodeEmbedding
            pos_embedding (dict): config of position embedding
            relative_pos_embedding (dict): config of relative position embedding
            textual_relation_module (dict): relation module for textual context, e.g. BertEncoder
            mode(int): fusion mode
            with_visual(bool): whether to use visual feature
        """
        super(MultiModalContextModulePlusPlus, self).__init__()

        self.mode = mode
        self.with_visual = with_visual
        self.textual_embedding = build_embedding(textual_embedding)

        # build position embedding
        if pos_embedding is not None:
            self.pos_embedding = build_embedding(pos_embedding)

        self.visual_ap = nn.AdaptiveAvgPool2d((1, 1))
        self.layernorm = torch.nn.LayerNorm(textual_embedding.get('sentence_embedding').get('embedding_dim', 128))

        head_num = None
        if textual_relation_module is not None:
            raw_ber_config = textual_relation_module.get('config', None)
            assert raw_ber_config is not None
            head_num = raw_ber_config.get('num_attention_heads', 1)

            # bert parameters
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
            # only one self-att
            if self.mode in [0, 1]:
                textual_relation_module['config'] = bert_config
                self.textual_relation_module = build_connect(textual_relation_module)
            elif self.mode == 2:
                textual_relation_module['config'] = bert_config
                visual_relation_module = copy.deepcopy(textual_relation_module)
                multimodal_relation_module = copy.deepcopy(textual_relation_module)
                self.textual_relation_module = build_connect(textual_relation_module)
                self.visual_relation_module = build_connect(visual_relation_module)
                self.multimodal_relation_module = build_connect(multimodal_relation_module)

        # relative position embedding
        if relative_pos_embedding is not None:
            if head_num is not None:
                relative_pos_embedding['embedding_dim'] = head_num
            self.relative_pos_embedding = build_embedding(relative_pos_embedding)

    @property
    def with_pos_embedding(self):
        """
        Returns:
            Determine the model with the position embedding or not
        """
        return hasattr(self, 'pos_embedding') and self.pos_embedding is not None

    @property
    def with_textual_relation_module(self):
        """
        Returns:
            Determine the model with the textual relation module or not
        """
        return hasattr(self, 'textual_relation_module') and self.textual_relation_module is not None

    @property
    def with_relative_pos_embedding(self):
        """
        Returns:
            Determine the model with the relative position embedding or not
        """
        return hasattr(self, 'relative_pos_embedding') and self.relative_pos_embedding is not None

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if pretrained is None:
            return

    def pack_batch(self,
                   feat_all,
                   pos_feat,
                   img_meta,
                   info_labels,
                   bieo_labels=None,
                   gt_texts=None,
                   char_nums=None):
        """ Forward computation

        Args:
            feat_all (list(Tensor)):
                0: visual features, in shape of N x C x H x W, where N is all text bboxes in a batch
                1: textual features, in shape of N x L x C, where L is max length of per text bbox
            pos_feat (list(Tensor)): box Tensor for each sample in a batch, e.g. B x 4
            img_meta (list(dict)): img_metas for each sample in a batch
            info_labels (list(Tensor)): category labels for all text bboxes
            bieo_labels (list(list(Tensor))): category labels for each characters in all bboxes if exist
            gt_texts (list(list(str))): ground truth texts
            char_nums (list(list(int))): char numbers for every texts

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
        Returns:
            list(str): texts to same length, in Max_N
        Returns:
            list(int): char numbers to same length, in Max_N
        """
        img_feat_all = feat_all
        max_length = max([per_b.size(0) for per_b in pos_feat])
        ## 0: visual feat, 1: textual feat
        batched_img_feat = [[], []]
        batched_pos_feat = []
        batched_img_label = []
        batched_img_bieo_label = []
        batched_gt_texts = []
        batched_char_nums = []
        last_idx = 0

        # pack
        for _ in range(len(pos_feat)):
            # visual feat
            b_s = pos_feat[_].size(0)
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
            per_pos_feat = pos_feat[_]
            image_shape_h, image_shape_w = img_meta[_]['img_shape'][:2]
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
                per_label = info_labels[_]
                img_feat_size = list(per_label.size())
                img_feat_size[0] = max_length - b_s
                batched_img_label.append(
                    torch.cat((per_label, per_label.new_full(img_feat_size, 255)), 0))

            # bieo labels
            if bieo_labels is not None:
                per_label = copy.deepcopy(bieo_labels[_])
                per_label = torch.tensor(per_label, dtype=torch.long).to(img_feat.device)

                img_feat_size = list(per_label.size())
                img_feat_size[0] = max_length - b_s
                batched_img_bieo_label.append(
                    torch.cat((per_label, per_label.new_full(img_feat_size, 255)), 0))

            # gt_texts
            if gt_texts is not None:
                per_gt_text = copy.deepcopy(gt_texts[_])
                pad_num = max_length - len(per_gt_text)
                per_gt_text.extend(['[PAD]'] * pad_num)
                batched_gt_texts.append(per_gt_text)

            # char_nums
            if char_nums is not None:
                per_gt_text = copy.deepcopy(char_nums[_])
                # if per_gt_text == 0:
                #     print(img_meta[_]['filename'])
                pad_num = max_length - len(per_gt_text)
                per_gt_text.extend([0] * pad_num)
                batched_char_nums.append(per_gt_text)

            last_idx += b_s

        return batched_img_feat, batched_pos_feat, batched_img_label, batched_img_bieo_label, batched_gt_texts, batched_char_nums

    def forward(self,
                info_feat_list,
                pos_feat,
                img_metas,
                info_labels=None,
                bieo_labels=None,
                gt_texts=None,
                char_nums=None,
                return_dict=False):
        """ Forward computation

        Args:
            info_feat_list (list(Tensor)):
                0: visual features, in shape of N x C x H x W, where N is all text bboxes in a batch
                1: textual features, in shape of N x L x C, where L is max length of per text bbox
            pos_feat (list(Tensor)): box Tensor for each sample in a batch, e.g. B x 4
            img_metas (list(dict)): img_metas for each sample in a batch
            info_labels (list(Tensor)): category labels for all text bboxes
            bieo_labels (list(list(Tensor))): category labels for each characters in all bboxes if exist
            gt_texts (list(list(str))): ground truth texts
            char_nums (list(list(int))): char numbers for every texts
            return_dict(bool): whether to return as dict

        Returns:
            Tensor: fused feature maps, in shape of [B x Max_N x C]
        Returns:
            list(Tensor): category labels for bboxes in each sample
        Returns:
            list(Tensor): bieo label Tensor to same length, in B x Max_N x L
        Returns:
            list(Tensor): bert embedding Tensor to same length, in B x Max_N x L
        """
        # pack data in a batch to same length
        batched_img_feat, batched_pos_feat, batched_img_label, batched_img_bieo_label, batched_gt_texts, batched_char_nums = self.pack_batch(
            info_feat_list,
            pos_feat=pos_feat,
            img_meta=img_metas,
            info_labels=info_labels,
            bieo_labels=bieo_labels,
            gt_texts=gt_texts,
            char_nums=char_nums)
        # textual embedding
        origin_textual_context = batched_img_feat[1]
        textual_context, bert_token_embeddings = self.textual_embedding(origin_textual_context, batched_gt_texts,
                                                                        batched_char_nums)

        # layout embedding
        position_embedding = torch.zeros_like(textual_context)
        if self.with_pos_embedding:
            position_embedding = self.pos_embedding(torch.stack(batched_pos_feat, 0))

        relative_pos_embedding = None
        # relative pos embedding
        if self.with_relative_pos_embedding:
            relative_pos_embedding = self.relative_pos_embedding(torch.stack(batched_pos_feat, 0))
            relative_pos_embedding = relative_pos_embedding.permute(0, 3, 1, 2).contiguous()

        # visual embedding
        origin_visual_context = batched_img_feat[0]
        visual_feat = torch.cat(origin_visual_context, 0)
        visual_feat = self.visual_ap(visual_feat).squeeze(2).squeeze(2)
        visual_feat = visual_feat.view(textual_context.size(0), -1, visual_feat.size(-1))

        if self.mode == 0:
            # 0: visual + self_att(layout + textual)
            node_embedding = self.layernorm(position_embedding + textual_context)
            # self_att
            head_mask = [None] * self.infor_bert_config.num_hidden_layers
            tmp_batched_pos_bboxes = torch.stack(batched_pos_feat, 0)
            sentence_mask = (torch.sum(torch.abs(tmp_batched_pos_bboxes), dim=-1) != 0).float()
            sentence_mask = sentence_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            sentence_mask = (1.0 - sentence_mask) * -10000.0
            all_bert_outputs = self.textual_relation_module(node_embedding,
                                                            attention_mask=sentence_mask[:, None, None, :],
                                                            head_mask=head_mask,
                                                            relative_alpha=relative_pos_embedding)
            node_embedding = all_bert_outputs[0]
            # + visual
            multimodal_contenxt = node_embedding + visual_feat
        elif self.mode == 1:
            # 1: self_att(layout + visual + textual)
            if self.with_visual:
                node_embedding = self.layernorm(position_embedding + textual_context + visual_feat)
            else:
                node_embedding = self.layernorm(position_embedding + textual_context)
            # self_att
            head_mask = [None] * self.infor_bert_config.num_hidden_layers
            tmp_batched_pos_bboxes = torch.stack(batched_pos_feat, 0)
            sentence_mask = (torch.sum(torch.abs(tmp_batched_pos_bboxes), dim=-1) != 0).float()
            sentence_mask = sentence_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            sentence_mask = (1.0 - sentence_mask) * -10000.0
            all_bert_outputs = self.textual_relation_module(node_embedding,
                                                            attention_mask=sentence_mask[:, None, None, :],
                                                            head_mask=head_mask,
                                                            relative_alpha=relative_pos_embedding)
            multimodal_contenxt = all_bert_outputs[0]
        elif self.mode == 2:
            # 2: self_att( self_att(layout + visual) + self_att(layout + textual))
            # shared mask
            head_mask = [None] * self.infor_bert_config.num_hidden_layers
            tmp_batched_pos_bboxes = torch.stack(batched_pos_feat, 0)
            sentence_mask = (torch.sum(torch.abs(tmp_batched_pos_bboxes), dim=-1) != 0).float()
            sentence_mask = sentence_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            sentence_mask = (1.0 - sentence_mask) * -10000.0
            # self_att(layout + visual)
            visual_node_embedding = self.layernorm(position_embedding + visual_feat)
            visual_node_embedding = self.visual_relation_module(visual_node_embedding,
                                                                attention_mask=sentence_mask[:, None, None, :],
                                                                head_mask=head_mask,
                                                                relative_alpha=relative_pos_embedding)[0]
            # self_att(layout + textual)
            textual_node_embedding = self.layernorm(position_embedding + textual_context)
            textual_node_embedding = self.visual_relation_module(textual_node_embedding,
                                                                 attention_mask=sentence_mask[:, None, None, :],
                                                                 head_mask=head_mask,
                                                                 relative_alpha=relative_pos_embedding)[0]
            # self_att(visual_node_embedding + textual_node_embedding)
            multimodal_contenxt = self.layernorm(textual_node_embedding + visual_node_embedding + position_embedding)
            multimodal_contenxt = self.visual_relation_module(multimodal_contenxt,
                                                              attention_mask=sentence_mask[:, None, None, :],
                                                              head_mask=head_mask,
                                                              relative_alpha=relative_pos_embedding)[0]
        else:
            raise NotImplementedError

        if return_dict:
            return {
                'multimodal_contenxt': multimodal_contenxt,
                'batched_img_label': batched_img_label,
                'batched_img_bieo_label': batched_img_bieo_label,
                'bert_token_embeddings': bert_token_embeddings,
                'batched_gt_texts': batched_gt_texts,
                'batched_pos_feat': batched_pos_feat
            }
        else:
            return multimodal_contenxt, batched_img_label, batched_img_bieo_label, bert_token_embeddings
