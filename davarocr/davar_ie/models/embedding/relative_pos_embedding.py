"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    relative_pos_embedding.py
# Abstract       :    relative position embedding module

# Current Version:    1.0.0
# Date           :    2022-11-22
#########################################################################
"""
import torch
import torch.nn as nn
from davarocr.davar_common.models.builder import EMBEDDING


@EMBEDDING.register_module()
class RelativePositionEmbedding2D(nn.Module):
    """Relative 2D Postion Embedding layer. """

    def __init__(self,
                 max_position_embeddings=128,
                 embedding_dim=128, ):
        """
        Args:
            max_position_embeddings (int): max normalized input dimension (similar to vocab_size).
            embedding_dim (int): size of embedding vector.
        """
        super(RelativePositionEmbedding2D, self).__init__()

        self.max_position_embeddings = max_position_embeddings
        self.pos_embedding_dim = embedding_dim

        # [-C, C]
        self.x_embedding = nn.Embedding(self.max_position_embeddings * 2, self.pos_embedding_dim)
        self.y_embedding = nn.Embedding(self.max_position_embeddings * 2, self.pos_embedding_dim)

    def init_weights(self, pretrained=None):
        """
            init layer weights.
        """
        if pretrained is None:
            return

    def forward(self, gt_bboxes):
        """
        Forward process.
        Args:gt_bboxes: shape of [BxLengthx4],
            typical the top-left and bottom-right points.
            Normalized (0~1)

        Returns:
           tensor: relative position embedding
        """
        # normalize to max_position_embeddings.
        gt_bboxes = gt_bboxes * self.max_position_embeddings
        # average x/ y coordinates
        avg_x = (gt_bboxes[:, :, 0] + gt_bboxes[:, :, 2]) / 2.
        avg_y = (gt_bboxes[:, :, 1] + gt_bboxes[:, :, 3]) / 2.
        # relative x/ y
        relative_x = avg_x[:, :, None] - avg_x[:, None, :] + self.max_position_embeddings
        relative_y = avg_y[:, :, None] - avg_y[:, None, :] + self.max_position_embeddings
        # clamp
        relative_x = torch.clamp(relative_x, 0,
                                 self.max_position_embeddings * 2 - 1).long()
        relative_y = torch.clamp(relative_y, 0,
                                 self.max_position_embeddings * 2 - 1).long()

        relative_x = self.x_embedding(relative_x)
        relative_y = self.y_embedding(relative_y)

        return (relative_x + relative_y)
