"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    embedding.py
# Abstract       :    torch.nn.Embedding function encapsulation.

# Current Version:    1.0.0
# Date           :    2021-05-20
######################################################################################################
"""
from torch import nn
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import EMBEDDING
from davarocr.davar_common.utils import get_root_logger


@EMBEDDING.register_module()
class Embedding(nn.Module):
    """ Embedding layer. Raw implementation: nn.Embedding(vocab_size, embedding_dim)"""
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 drop_out=0.):
        """
        Args:
            vocab_size (int): size of vocabulary.
            embedding_dim (int): dim of input features
            drop_out (float): drop_out ratio if required.
        """
        super().__init__()
        self.drop_out = drop_out
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop_out_layer = nn.Dropout(self.drop_out)

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Embedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input_feature):
        """ Forward computation

        Args:
            input_feature (Tensor): in shape of [B x N x L]
        Returns:
            Tensor: in shape of [B x N x L x D], where D is the embedding_dim.
        """
        embed_vector = self.embedding(input_feature)
        embed_vector = self.drop_out_layer(embed_vector)
        return embed_vector
