"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    gcn_head.py
# Abstract       :    gcn classification head.

# Current Version:    1.0.0
# Date           :    2021-12-13
######################################################################################################
"""
import torch
from torch import nn
import numpy as np

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmcv.runner import auto_fp16, force_fp32, load_checkpoint

from davarocr.davar_common.models import build_connect, build_embedding
from davarocr.davar_ie.models import BertConfig
from davarocr.davar_common.utils import get_root_logger


@HEADS.register_module()
class GCNHead(nn.Module):
    """Head with GCN module."""

    def __init__(self,
                 in_channels=256,
                 num_fcs=0,
                 roi_feat_size=0,
                 fc_out_channels=0,
                 pos_embedding=None,
                 gcn_module=None,
                 dropout_ratio=0.,
                 with_line_cls=True,
                 num_classes=None,
                 loss_line_cls=None,
                 with_short_cut=False,
                 local_size=2048):
        super().__init__()
        output_channels = in_channels
        self.in_channels = in_channels

        self.num_fcs = num_fcs
        self.roi_feat_size = roi_feat_size
        self.fc_out_channels = fc_out_channels
        self.cls_fcs = self._add_fc_branch(self.num_fcs, self.in_channels)

        self.relu = nn.ReLU(inplace=True)

        if self.fc_out_channels > 0:
            output_channels = self.fc_out_channels

        # position embedding
        if pos_embedding is not None:
            assert pos_embedding.embedding_dim == output_channels
            self.position_embedding = build_embedding(pos_embedding)
            self.layernorm = torch.nn.LayerNorm(pos_embedding.embedding_dim)

        # gcn module
        if gcn_module is not None:
            # bertencoder
            raw_ber_config = gcn_module.get('config', None)
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
            gcn_module.update(config=bert_config)
            self.bert_config = bert_config
            self.gcn_module = build_connect(gcn_module)
            output_channels = bert_config.hidden_size

        self.dropout = nn.Dropout(dropout_ratio)

        # line cls
        if with_line_cls and loss_line_cls is not None:
            self.line_cls = nn.Linear(output_channels, num_classes + 1)
            self.loss_line_cls = build_loss(loss_line_cls)

        self.with_short_cut = with_short_cut
        self.local_size = local_size

    def _add_fc_branch(self,
                        num_branch_fcs,
                        in_channels):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            last_layer_dim = last_layer_dim * self.roi_feat_size * self.roi_feat_size
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def init_weights(self, pretrained):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("GCN_Head:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    @property
    def with_line_cls(self):
        """

        Returns:
            bool: Determine the model with the line_cls or not
        """
        return hasattr(self, 'line_cls') and self.line_cls is not None

    @property
    def with_pos_embedding(self):
        """

        Returns:
            bool: Determine the model with the position_embedding or not
        """
        return hasattr(self, 'position_embedding') and self.position_embedding is not None

    @property
    def with_gcn_module(self):
        """

        Returns:
            bool: Determine the model with the gcn_module or not
        """
        return hasattr(self, 'gcn_module') and self.gcn_module is not None

    def pack_batch(self,
                   line_feats,
                   rois,
                   img_metas):
        """Pack input to same length, required by the following GCM module.

        Args:
        	line_feats (Tensor): input feature in shape NxD, where N is total number of texts in a batch, and D is the
        	    dimension.
        	rois (Tensor): roi infor in shape Nx5, where N is the same with line_feats.
        	img_metas (List): img_meta infor corresponds to samples in a batch.

        Returns:
        	Tensor: batched feat in shape BxMxD, where B is the batchsize, M is the max number of texts in a batch and
        	    D is the feature dimension.
        	Tensor: batched normalized bbox coordinates with [0, 0, 0, 0] padded. In shape of BxMx4
        	List: pad number, used to filter non-valid padding texts.

        """
        gt_bboxes = []
        last_idx = 0
        for _ in range(len(img_metas)):
            b_s = (rois[:, 0] == _).int().sum().item()
            gt_bboxes.append(rois[last_idx: last_idx + b_s, 1:])
            last_idx += b_s

        max_length = max([per_b.size(0) for per_b in gt_bboxes])
        batched_img_feat = []
        batched_pos_feat = []
        non_valid_num = []
        last_idx = 0

        # pack
        for _ in range(len(gt_bboxes)):
            # visual feat
            b_s = gt_bboxes[_].size(0)

            img_feat_size = list(line_feats.size())
            img_feat_size[0] = max_length - b_s
            batched_img_feat.append(
                torch.cat((line_feats[last_idx: last_idx + b_s], line_feats.new_full(img_feat_size, 0)), 0))

            # pos feat
            per_pos_feat = gt_bboxes[_]
            image_shape_h, image_shape_w = img_metas[_]['img_shape'][:2]
            per_pos_feat_expand = per_pos_feat.new_full((per_pos_feat.size(0), 4), 0)
            per_pos_feat_expand[:, 0] = per_pos_feat[:, 0]
            per_pos_feat_expand[:, 1] = per_pos_feat[:, 1]
            per_pos_feat_expand[:, 2] = per_pos_feat[:, 2]
            per_pos_feat_expand[:, 3] = per_pos_feat[:, 3]
            img_feat_size = list(per_pos_feat_expand.size())
            img_feat_size[0] = max_length - b_s

            # normalize
            per_pos_feat_expand[:, ::2] = \
                per_pos_feat_expand[:, ::2] / image_shape_w
            per_pos_feat_expand[:, 1::2] = \
                per_pos_feat_expand[:, 1::2] / image_shape_h

            batched_pos_feat.append(
                torch.cat((per_pos_feat_expand, per_pos_feat_expand.new_full(img_feat_size, 0)), 0))

            last_idx += b_s

            non_valid_num.append(max_length-b_s)
        return torch.stack(batched_img_feat, 0), torch.stack(batched_pos_feat, 0), non_valid_num

    def _relation_forward(self, batched_line_feats, batched_pos_feat):
        """Relation modeling forward function.

        Args:
            batched_line_feats (Tensor): node feature in shape BxNxD, where B is the batchsize, N is total number of
                texts in a batch, and D is the dimension.
            batched_pos_feat (Tensor): node pos feature in shape BxNx4, where B and N are the same with
                batched_line_feats.

        Returns:
            Tensor: output feature, in the same shape iwth batched_line_feats.
        """
        if self.with_gcn_module:
            head_mask = [None] * self.bert_config.num_hidden_layers
            sentence_mask = (torch.sum(torch.abs(batched_pos_feat), dim=-1) != 0).float()
            sentence_mask = sentence_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            sentence_mask = (1.0 - sentence_mask) * -10000.0
            batched_line_feats = self.gcn_module(batched_line_feats,
                                                 attention_mask=sentence_mask[:, None, None, :],
                                                 head_mask=head_mask)[0]

        return batched_line_feats

    @auto_fp16()
    def forward(self, line_feats, rois, img_metas):
        """Forward process.

        Args:
            line_feats (Tensor): input feature in shape NxD, where N is total number of texts in a batch, and D is the
                dimension.
            rois (Tensor): roi infor in shape Nx5, where N is the same with line_feats.
            img_metas (List): img_meta infor corresponds to samples in a batch.

        Returns:
            Tensor: output feature (optional of gcn module), in the shape of NxD.
        """
        line_feats_copy = line_feats

        # batch max length
        max_length = 0
        for _ in range(len(img_metas)):
            b_s = (rois[:, 0] == _).int().sum().item()
            if b_s > max_length:
                max_length = b_s

        if line_feats.ndim == 4:
            line_feats = line_feats.flatten(1)

        batched_line_feats, batched_pos_feat, non_valid_num = self.pack_batch(line_feats, rois, img_metas)

        for fc in self.cls_fcs:
            batched_line_feats = self.relu(fc(batched_line_feats))

        if self.with_pos_embedding:
            batched_pos_feat_emb = self.position_embedding(batched_pos_feat)
            batched_line_feats = self.dropout(self.layernorm(batched_pos_feat_emb + batched_line_feats))

        # inference phase, handle too long documents using local window
        if not self.training:
            true_length = batched_line_feats.size(1)
            feat_dim = batched_line_feats.size(-1)
            pad_length = int(np.ceil(true_length / self.local_size) * self.local_size - true_length)
            if pad_length > 0:
                img_feat_size = list(batched_line_feats.size())
                img_feat_size[1] = pad_length
                pad_line_feats = torch.cat((batched_line_feats, batched_line_feats.new_full(img_feat_size, 0)), 1)
                img_feat_size = list(batched_pos_feat.size())
                img_feat_size[1] = pad_length
                pad_pos_feats = torch.cat((batched_pos_feat, batched_pos_feat.new_full(img_feat_size, 0)), 1)
                pad_line_feats = pad_line_feats.view(-1, self.local_size, pad_line_feats.size(-1))
                pad_pos_feats = pad_pos_feats.view(-1, self.local_size, pad_pos_feats.size(-1))
                result = []
                # to avoid memory limit
                for idx in range(pad_line_feats.size(0)):
                    result.append(self._relation_forward(pad_line_feats[idx, :, :][None, :, :],
                                                         pad_pos_feats[idx, :, :][None, :, :]))
                batched_line_feats = torch.cat(result, 0).view(1, -1, feat_dim)[:, :true_length, :]
            else:
                batched_line_feats = self._relation_forward(batched_line_feats, batched_pos_feat)
        else:
            batched_line_feats = self._relation_forward(batched_line_feats, batched_pos_feat)

        valid_batch_output = []
        for idx, per in enumerate(non_valid_num):
            valid_batch_output.append(batched_line_feats[idx, :batched_line_feats.size(1)-per])

        line_cls = torch.cat(valid_batch_output, 0)

        if self.with_short_cut:
            line_cls += line_feats_copy

        if self.with_line_cls:
            line_cls = self.line_cls(line_cls)

        return line_cls

    def get_targets(self, gt_labels):
        """Return target labels.

        Args:
        	gt_labels (list): gt_label for each sample in a batch.

        Returns:
        	Tensor: concatenation gt_labels.

        """
        return torch.cat(gt_labels, 0).long()

    @force_fp32(apply_to=('line_cls_score'))
    def loss(self, line_cls_score, line_cls_targets, prefix=''):
        """Loss computation.

        Args:
        	line_cls_score (Tensor): pred result, in the shape of NxC, where N is the total number, C is the number of
        	    Classes.
        	line_cls_targets (Tensor): gt labels, in the shape of N.
        	prefix (str): loss key prefix, to differentiate with other losses.

        Returns:
        	dict: computed losses.

        """
        losses = dict()
        if line_cls_score is not None:
            line_cls_score = line_cls_score.view(-1, line_cls_score.size(-1))
            line_cls_targets = line_cls_targets.view(-1)
            losses[prefix+'loss_line_cls'] = self.loss_line_cls(
                line_cls_score,
                line_cls_targets)
        losses['acc'] = accuracy(line_cls_score, line_cls_targets)
        return losses
