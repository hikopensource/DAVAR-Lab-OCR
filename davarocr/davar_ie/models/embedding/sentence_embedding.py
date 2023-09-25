"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    sentence_embedding.py
# Abstract       :    sentence embedding for each texts.

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""
import torch
from torch import nn
from mmcv.runner import load_checkpoint

from transformers import AutoModel, AutoTokenizer

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

@EMBEDDING.register_module()
class SentenceEmbeddingBertNew(nn.Module):
    """SentenceBertEmbedding layer"""
    def __init__(self,
                 auto_model_path,
                 embedding_dim=768,
                 freeze_params=True,
                 character_wise=False,
                 use_cls=True,
                 remap=False,
                 batch_max_length=None):
        """
        Args：
            auto_model_path: pretrained language model path
            vocab_size: size of the vocabulary
            embedding_dim: embedding dimension
            freeze_params: whether freeze the pretrained params
            character_wise: whether use cha character feature
            use_cls: whether to use cls segment
            remap: whether to remap
            batch_max_length: max sentence length

        Returns：
        """
        super(SentenceEmbeddingBertNew, self).__init__()

        assert auto_model_path is not None
        self.embedding_dim = embedding_dim
        self.automodel = AutoModel.from_pretrained(auto_model_path)
        if freeze_params:
            # freeze automodel params
            self._freeze_automodel()
        self.autotokenizer = AutoTokenizer.from_pretrained(auto_model_path)
        self.id_2_txt = {v: k for k, v in self.autotokenizer.vocab.items()}
        self.txt_2_id = {k: v for k, v in self.autotokenizer.vocab.items()}
        self.use_cls = use_cls
        self.remap = remap
        self.character_wise = character_wise
        self.batch_max_length = batch_max_length

    def init_weights(self, pretrained):
        """
            init segmentation head weights.
        """
        pass

    def forward(self, tmp_feature, gt_texts):
        """
        Args:
            tmp_feature: input features
            gt_texts: text ground truth

        Returns: feature embedding
        """
        # device infor
        device = tmp_feature.device

        # extract embedding
        bert_embeddings = []
        bert_token_embeddings = []
        for img_idx, per_img in enumerate(gt_texts):
            token_embeddings = torch.zeros((len(per_img), self.batch_max_length, self.embedding_dim), device=device)

            inputs = self.autotokenizer(per_img, return_tensors='pt', padding=True, truncation=True)
            inputs.to(device)
            outputs = self.automodel(**inputs)

            bert_embeddings.append(outputs[1][:, :self.embedding_dim])
            last_hidden_state = outputs[0]

            encodings = inputs.encodings

            for idx_b, per_encoding in enumerate(encodings):
                offsets = per_encoding.offsets
                attention_mask = per_encoding.attention_mask
                for idx_t in range(len(offsets)):
                    if attention_mask[idx_t] == 1 and offsets[idx_t][1] != 0:
                        start, end = offsets[idx_t]
                        token_embeddings[idx_b, start: end, :] = last_hidden_state[idx_b, idx_t, :self.embedding_dim]
            # vis_arr = token_embeddings[:, :, 0].cpu().numpy()
            bert_token_embeddings.append(token_embeddings)

        return torch.stack(bert_embeddings, 0), torch.stack(bert_token_embeddings, 0)

    def _freeze_automodel(self):
        """freeze parameters"""
        for name, param in self.automodel.named_parameters():
            param.requires_grad = False
