"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    node_embedding.py
# Abstract       :    node embedding for each bbox, consisting of textual and layout embeddings.

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
import torch
from torch import nn
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import EMBEDDING
from davarocr.davar_common.models.builder import build_embedding
from davarocr.davar_common.utils import get_root_logger


@EMBEDDING.register_module()
class NodeEmbedding(nn.Module):
    """NodeEmbedding (for each bbox). """
    def __init__(self,
                 pos_embedding=None,
                 merge_type='Sum',
                 dropout_ratio=0.1,
                 sentence_embedding=None
                 ):
        """
        Args:
            pos_embedding (dict): pos embedding module, e.g. PositionEmbedding2D
            merge_type (str): fusion type, e.g. 'Sum', 'Concat'
            dropout_ratio (float): dropout ratio of fusion features
            sentence_embedding (dict): sentence embedding module, e.g. SentenceEmbeddingCNN
        """
        super().__init__()

        # pos embedding
        self.pos_embedding = build_embedding(pos_embedding)
        self.pos_embedding_dim = pos_embedding.get('embedding_dim', 128)

        # sentence_embedding
        self.sentence_embedding = build_embedding(sentence_embedding)
        self.sentence_embedding_dim = sentence_embedding.get('embedding_dim', 128)

        # merge param
        self.merge_type = merge_type
        if self.merge_type == 'Sum':
            assert self.sentence_embedding_dim == self.pos_embedding_dim

            self.layernorm = nn.LayerNorm(self.pos_embedding_dim)
        elif self.merge_type == 'Concat':
            self.layernorm = nn.LayerNorm(self.sentence_embedding_dim + self.pos_embedding_dim)
        else:
            raise "Unknown merge type {}".format(self.merge_type)

        self.dropout = nn.Dropout(dropout_ratio)

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Node Embedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, recog_hidden, gt_bboxes):
        """ Forward computation

        Args:
            recog_hidden (list(Tensor)): textual feature maps, in shape of [N x L x C] x B
            gt_bboxes (Tensor): textual feature maps, in shape of [N x 4] x B
        Returns:
            Tensor: fused feature maps, in shape of [B x L x C]
        """
        # shape reform
        recog_hidden = torch.stack(recog_hidden, 0)
        gt_bboxes = torch.stack(gt_bboxes, 0)

        # sentence_embedding
        x_sentence = self.sentence_embedding(recog_hidden)

        # positiion embedding
        sum_position_embedding = self.pos_embedding(gt_bboxes)

        # feature merge
        if self.merge_type == 'Sum':
            x_sentence = x_sentence + sum_position_embedding
        elif self.merge_type == 'Concat':
            x_sentence = torch.cat((x_sentence, sum_position_embedding), -1)

        x_sentence = self.layernorm(x_sentence)
        x_sentence = self.dropout(x_sentence)

        return x_sentence
