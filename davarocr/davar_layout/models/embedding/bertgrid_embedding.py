"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    bertgrid_embedding.py
# Abstract       :    generate bertgrid embedding feature map.

# Current Version:    1.0.0
# Date           :    2021-12-13
######################################################################################################
"""
import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import EMBEDDING
from davarocr.davar_common.utils import get_root_logger


@EMBEDDING.register_module()
class BERTgridEmbedding(nn.Module):
    """Generate bertgrid embedding using tokenizers from pretrained language models.
    """

    def __init__(self,
                 auto_model_path,
                 embedding_dim=64,
                 batch_max_num=128):
        """
        Args：
            auto_model_path (str): path to pretrained language model (e.g. BERT)
            embedding_dim (int): dim of input features
            batch_max_num (int): the max num of texts in a batch due to memory limit, default 128.
    """
        super().__init__()

        assert auto_model_path is not None
        self.batch_max_num = batch_max_num
        self.embedding_dim = embedding_dim
        self.autotokenizer = AutoTokenizer.from_pretrained(auto_model_path)
        self.embedding = nn.Embedding(self.autotokenizer.vocab_size, embedding_dim)

    def init_weights(self, pretrained):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("BERTgridEmbedding:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            return
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self,
                img,
                gt_bboxes,
                gt_texts):
        """ Forward computation

        Args:
            img (Tensor): in shape of [B x C x H x W].
            gt_bboxes (list(Tensor)): bboxes for each text line in each image.
            gt_texts (list(list)): text contents for each image.
        Returns:
            Tensor: generated grid embedding maps in shape of [B x D x H x W], where D is the embedding_dim.
        """
        # restore feature map
        device = img.device
        batch_b, _, batch_h, batch_w = img.size()

        chargrid_map = torch.zeros((batch_b, 1, batch_h, batch_w), dtype=torch.int64).to(device)

        for iter_b in range(batch_b):
            per_img_texts = gt_texts[iter_b]
            if per_img_texts:
                gt_bboxes_arr = gt_bboxes[iter_b].cpu().numpy()

                ids = self.autotokenizer(per_img_texts)['input_ids']
                short_length_w = min(len(ids), gt_bboxes_arr.shape[0])

                for iter_b_l in range(short_length_w):
                    per_line_ids = ids[iter_b_l]
                    w_start, h_start, w_end, h_end = gt_bboxes_arr[iter_b_l].round().astype(np.int).tolist()
                    short_length_c = max(len(per_line_ids)-2, 0)

                    if short_length_c > 0:
                        span = (w_end - w_start) / short_length_c
                        for token_idx in range(1, 1+short_length_c):
                            per_id = per_line_ids[token_idx]
                            chargrid_map[iter_b, 0, h_start:h_end,
                            int(w_start + (token_idx - 1) * span): int(w_start + token_idx * span)] = per_id

        chargrid_map = self.embedding(chargrid_map).squeeze(1).permute(0, 3, 1, 2).contiguous()
        return chargrid_map
