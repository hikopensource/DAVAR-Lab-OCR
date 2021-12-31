"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    sentencegrid_embedding.py
# Abstract       :    generate sentencegrid embedding feature map.

# Current Version:    1.0.0
# Date           :    2021-12-13
######################################################################################################
"""
from torch import nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import EMBEDDING
from davarocr.davar_common.utils import get_root_logger


@EMBEDDING.register_module()
class SentencegridEmbedding(nn.Module):
    """Generate sentence embedding using pretrained language model, e.g. BERT
    """

    def __init__(self,
                 auto_model_path,
                 embedding_dim=768,
                 freeze_params=True,
                 batch_max_num=128):
        """
        Args：
            auto_model_path (str): path to pretrained language model (e.g. BERT)
            embedding_dim (int): dim of input features
            freeze_params (boolean): whether to freeze params of pretrained language model, default to True.
            batch_max_num (int): the max num of texts in a batch due to memory limit, default 128.
    """
        super().__init__()

        assert auto_model_path is not None
        self.batch_max_num = batch_max_num
        self.embedding_dim = embedding_dim
        self.automodel = AutoModel.from_pretrained(auto_model_path)
        self.autotokenizer = AutoTokenizer.from_pretrained(auto_model_path)
        if freeze_params:
            # freeze automodel params
            self._freeze_automodel()

    def init_weights(self, pretrained):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            logger.info("SentenceEmbedding:")
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
        # generate feature map
        device = img.device
        batch_b, _, batch_h, batch_w = img.size()
        chargrid_map = img.new_full((batch_b, self.embedding_dim, batch_h, batch_w), 0)

        for iter_b in range(batch_b):
            per_img_texts = gt_texts[iter_b]
            start_idx = 0
            while start_idx < len(per_img_texts):
                max_length = min(start_idx+self.batch_max_num, len(per_img_texts))
                per_batch_texts = per_img_texts[start_idx: max_length]
                inputs = self.autotokenizer(per_batch_texts, return_tensors='pt', padding=True, truncation=True)
                inputs.to(device)
                outputs = self.automodel(**inputs)
                pooler_output = outputs['pooler_output']

                valid_num = min(max_length-start_idx, pooler_output.size(0))
                for iter_b_l in range(valid_num):
                    w_start, h_start, w_end, h_end = gt_bboxes[iter_b][
                        start_idx + iter_b_l].cpu().numpy().round().astype(np.int).tolist()
                    chargrid_map[iter_b, :, h_start: h_end, w_start: w_end] = pooler_output[iter_b_l,
                                                                              :self.embedding_dim, None, None]
                start_idx += max_length

        return chargrid_map

    def _freeze_automodel(self):
        """Freeze params inside this model.
        """
        for _, param in self.automodel.named_parameters():
            param.requires_grad = False
