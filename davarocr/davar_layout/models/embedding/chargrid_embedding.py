"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    chargrid_embedding.py
# Abstract       :    generate chargrid embedding feature map.

# Current Version:    1.0.0
# Date           :    2021-12-13
######################################################################################################
"""
import numpy as np
import torch
from torch import nn
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import EMBEDDING
from davarocr.davar_common.utils import get_root_logger


@EMBEDDING.register_module()
class ChargridEmbedding(nn.Module):
    """Generate chargrid embedding feature map.
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 drop_out=0.):
        """
        Args：
            vocab_size (int): size of vocabulary.
            embedding_dim (int): dim of input features
            drop_out (float): drop_out ratio if required.
        """
        super().__init__()

        self.drop_out = drop_out
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop_out_layer = nn.Dropout(self.drop_out)

    def init_weights(self, pretrained):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("ChargridEmbedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, img, gt_ctexts, gt_cbboxes):
        """ Forward computation

        Args:
            img (Tensor): in shape of [B x N x L]
        Returns:
            Tensor: in shape of [B x N x L x D], where D is the embedding_dim.
        """
        # restore feature map
        device = img.device
        batch_b, _, batch_h, batch_w = img.size()

        chargrid_map = torch.zeros((batch_b, 1, batch_h, batch_w), dtype=torch.int64).to(device)
        for iter_b in range(batch_b):
            per_input_ids = gt_ctexts[iter_b]
            short_length_w = min(len(per_input_ids), len(gt_cbboxes[iter_b]))

            for iter_b_l in range(short_length_w):
                per_line_ids = per_input_ids[iter_b_l]
                per_line_coords = gt_cbboxes[iter_b][iter_b_l].round().astype(np.int)
                # per_line_coords = gt_cbboxes[iter_b][iter_b_l].astype(np.int)

                short_length_c = min(len(per_line_ids), per_line_coords.shape[0])

                for token_idx in range(short_length_c):
                    per_id = per_line_ids[token_idx]
                    w_start, h_start, w_end, h_end = per_line_coords[token_idx].tolist()
                    chargrid_map[iter_b, 0, h_start:h_end, w_start: w_end] = per_id

        chargrid_map = self.embedding(chargrid_map).squeeze(1).permute(0, 3, 1, 2).contiguous()
        return chargrid_map
