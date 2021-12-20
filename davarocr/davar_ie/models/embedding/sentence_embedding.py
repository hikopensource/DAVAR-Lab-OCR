"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    sentence_embedding.py
# Abstract       :    sentence embedding for each texts.

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
class SentenceEmbeddingCNN(nn.Module):
    """SentenceEmbeddingCNN (for each text)."""
    def __init__(self,
                 embedding_dim,
                 output_dim=None,
                 kernel_sizes=None):
        """
        Args:
            embedding_dim (int): dim of input features
            output_dim (int or None): dim of output features, if not specified, use embedding_dim as default
            kernel_sizes (list(int): multiple kernels used in CNN to extract sentence embeddings
        """
        super().__init__()
        assert kernel_sizes is not None and isinstance(kernel_sizes, list)
        self.kernel_sizes = kernel_sizes
        self.embedding_dim = embedding_dim
        self.output_dim = self.embedding_dim if output_dim is None else output_dim

        # parallel cnn
        self.sentence_cnn_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=_),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)) for _ in kernel_sizes])
        # fc projection
        self.sentence_input_proj = nn.Linear(
            self.embedding_dim * len(kernel_sizes), self.output_dim)
        self.sentence_input_proj_relu = nn.ReLU()

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("Sentence Embedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input_feat, char_nums=None):
        """ Forward computation

        Args:
            input_feat (list(Tensor)): textual feature maps, in shape of [B x N x L x C]
            char_nums (list(int)): valid char nums in each text.
        Returns:
            Tensor: fused feature maps, in shape of [B x N x C]
        """
        feat_x = input_feat
        i_b, i_l, i_n, i_d = feat_x.size()
        img_feat = feat_x.view(-1, i_n, i_d).permute(0, 2, 1)

        conv_feat = []
        for per_cnn in self.sentence_cnn_conv:
            conv_feat.append(per_cnn(img_feat))

        img_feat = torch.cat(conv_feat, 1)
        img_feat = img_feat.squeeze(2).view(i_b, i_l, img_feat.size(1))
        x_sentence = self.sentence_input_proj_relu(self.sentence_input_proj(img_feat))
        return x_sentence
