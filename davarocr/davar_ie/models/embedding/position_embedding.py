"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    position_embedding.py
# Abstract       :    position embedding for each bbox.

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
import torch
from torch import nn
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import EMBEDDING
from davarocr.davar_common.utils import get_root_logger


@EMBEDDING.register_module()
class PositionEmbedding2D(nn.Module):
    """2D Postion Embedding layer. """
    def __init__(self,
                 max_position_embeddings=128,
                 embedding_dim=128,
                 width_embedding=False,
                 height_embedding=False,):
        """
        Args:
            max_position_embeddings (int): max normalized input dimension (similar to vocab_size).
            embedding_dim (int): size of embedding vector.
            width_embedding (bool): whether to include width embedding.
            height_embedding (bool): whether to include height embedding.
        """
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.pos_embedding_dim = embedding_dim

        self.x_embedding = nn.Embedding(self.max_position_embeddings, self.pos_embedding_dim)
        self.y_embedding = nn.Embedding(self.max_position_embeddings, self.pos_embedding_dim)
        self.width_embedding = None
        if width_embedding:
            self.width_embedding = nn.Embedding(self.max_position_embeddings, self.pos_embedding_dim)
        self.height_embedding = None
        if height_embedding:
            self.height_embedding = nn.Embedding(self.max_position_embeddings, self.pos_embedding_dim)

        # sum -> fc
        self.pos_input_proj = nn.Linear(self.pos_embedding_dim, self.pos_embedding_dim)
        self.pos_input_proj_relu = nn.ReLU()

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Position Embedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    @property
    def with_width_embedding(self):
        """

        Returns:
            Determine the model with the width_embedding or not
        """
        return hasattr(self, 'width_embedding') and self.width_embedding is not None

    @property
    def with_height_embedding(self):
        """

        Returns:
            Determine the model with the height_embedding or not
        """
        return hasattr(self, 'height_embedding') and self.height_embedding is not None

    def forward(self, gt_bboxes):
        """ Forward computation

        Args:
            gt_bboxes (Tensor): bboxes Tensor, in shape of [B x N x 4]
        Returns:
            Tensor: bboxes/ layout embeddings, in shape of [B x N x C]
        """
        # normalize to max_position_embeddings.
        gt_bboxes = torch.clamp((gt_bboxes * self.max_position_embeddings), 0, self.max_position_embeddings - 1).long()

        # top-left and bottom-right points embeddings
        left_position_embeddings = self.x_embedding(gt_bboxes[:, :, 0])
        upper_position_embeddings = self.y_embedding(gt_bboxes[:, :, 1])
        right_position_embeddings = self.x_embedding(gt_bboxes[:, :, 2])
        lower_position_embeddings = self.y_embedding(gt_bboxes[:, :, 3])
        sum_position_embedding = left_position_embeddings + upper_position_embeddings + right_position_embeddings +\
                                 lower_position_embeddings


        # include width embedding
        if self.with_width_embedding:
            sum_position_embedding += self.width_embedding(gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0])

        # include height embedding
        if self.with_height_embedding:
            sum_position_embedding += self.height_embedding(gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1])

        # sum & projection.
        sum_position_embedding = self.pos_input_proj_relu(self.pos_input_proj(sum_position_embedding))

        return sum_position_embedding
