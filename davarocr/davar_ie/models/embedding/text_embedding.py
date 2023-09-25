"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    text_embedding.py
# Abstract       :    text embedding for each texts.

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""
import torch
import torch.nn as nn
from davarocr.davar_common.models.builder import EMBEDDING
from davarocr.davar_common.models.builder import build_embedding

@EMBEDDING.register_module()
class TextualEmbedding(nn.Module):
    """
        node embedding implemntation in GCN

        Args:
            pos_embedding: pos embedding params.
            merge_type (Str): merge type of text embedding & position embedding,
                default, 'Sum'.
            dropout_ratio (float): dropout ratio used.
            sentence_embedding (dict): sentence_embedding params, contains
                'type' key and corresponding params.
    """
    def __init__(self,
                 merge_type='Gate',
                 dropout_ratio=0.1,
                 sentence_embedding=None,
                 sentence_embedding_bert=None,
                 fusion_phase='early'
                 ):
        super(TextualEmbedding, self).__init__()

        ## sentence_embedding
        self.sentence_embedding = build_embedding(sentence_embedding)
        self.sentence_embedding_dim = sentence_embedding.get('embedding_dim', 128)

        self.sentence_embedding_bert = sentence_embedding_bert
        if sentence_embedding_bert is not None:
            ## sentence_embedding_bert
            self.sentence_embedding_bert = build_embedding(sentence_embedding_bert)
            self.sentence_embedding_bert_dim = sentence_embedding_bert.get('embedding_dim', 128)

        # merge param
        self.merge_type = merge_type
        if self.with_sentence_embedding_bert:
            if self.merge_type == 'Sum':
                assert self.sentence_embedding_bert_dim == self.sentence_embedding_dim
                self.layernorm = torch.nn.LayerNorm(self.sentence_embedding_dim)
            elif self.merge_type == 'Concat':
                self.layernorm = torch.nn.LayerNorm(self.sentence_embedding_dim + self.sentence_embedding_bert_dim)
            elif self.merge_type == 'Gate':
                self.xx = nn.Linear(self.sentence_embedding_dim, self.sentence_embedding_dim)
                self.yy = nn.Linear(self.sentence_embedding_bert_dim, self.sentence_embedding_dim)
                self.xx_1 = nn.Linear(self.sentence_embedding_dim, self.sentence_embedding_dim)
                self.yy_1 = nn.Linear(self.sentence_embedding_bert_dim, self.sentence_embedding_dim)
            elif self.merge_type == 'GateNew':
                self.xx = nn.Linear(self.sentence_embedding_dim, self.sentence_embedding_dim)
                self.yy = nn.Linear(self.sentence_embedding_bert_dim, self.sentence_embedding_dim)
                self.yy_1 = nn.Linear(self.sentence_embedding_bert_dim, self.sentence_embedding_dim)
                self.relu = nn.ReLU(inplace=True)
            elif self.merge_type == 'SumNew':
                self.yy = nn.Linear(self.sentence_embedding_bert_dim, self.sentence_embedding_dim)
                self.relu = nn.ReLU(inplace=True)
                self.layernorm = torch.nn.LayerNorm(self.sentence_embedding_dim)
            else:
                raise "Unknown merge type {}, only support Sum Concat Gate GateNew SumNew".format(self.merge_type)

        self.dropout = nn.Dropout(dropout_ratio)
        self.fusion_phase = fusion_phase

    @property
    def with_sentence_embedding_bert(self):
        return hasattr(self, 'sentence_embedding_bert') and self.sentence_embedding_bert is not None

    def init_weights(self, pretrained=None):
        """
            init layer weights.
        """
        if pretrained is None:
            return

    def forward(self, recog_hidden, gt_texts=None, char_nums=None):
        """
        Args:
            recog_hidden(list(tensor)): char features from rcg
            gt_texts(list(str)): text ground truth
            char_nums (list(int)): valid char nums in each text.

        Returns: text embedding
        """
        ## shape reform
        recog_hidden = torch.stack(recog_hidden, 0)

        # cnn(rcg_hidden) + bert
        if self.fusion_phase == 'late':
            ## sentence_embedding
            x_sentence = self.sentence_embedding(recog_hidden, char_nums)

            x_sentence_token_embedding = None
            ## sentence_embedding_bert
            if self.with_sentence_embedding_bert:
                x_sentence_bert, x_sentence_token_embedding = self.sentence_embedding_bert(x_sentence, gt_texts)

            ## feature merge
            if self.merge_type == 'Sum':
                if self.with_sentence_embedding_bert:
                    segment_emb = x_sentence + x_sentence_bert
                    segment_emb = self.layernorm(segment_emb)
                else:
                    segment_emb = x_sentence
            elif self.merge_type == 'Concat':
                if self.with_sentence_embedding_bert:
                    segment_emb = torch.cat((x_sentence, x_sentence_bert), -1)
                    segment_emb = self.layernorm(segment_emb)
                else:
                    segment_emb = x_sentence
            elif self.merge_type == 'Gate':
                if self.with_sentence_embedding_bert:
                    b_, n_, d_ = x_sentence.size()
                    x_sentence = x_sentence.view(-1, d_).contiguous()
                    x_sentence_bert = x_sentence_bert.view(-1, x_sentence_bert.size(-1)).contiguous()
                    gate = torch.sigmoid(self.xx(x_sentence) + self.yy(x_sentence_bert))
                    new_info = torch.tanh(self.xx_1(x_sentence) + self.yy_1(x_sentence_bert))
                    segment_emb = x_sentence + (gate * new_info)
                    segment_emb = segment_emb.view(b_, n_, d_).contiguous()
                else:
                    segment_emb = x_sentence
        else:
            # cnn (rcg_hidden + bert)
            segment_emb = recog_hidden

            ret_sentence_token_embedding = None
            if self.with_sentence_embedding_bert:
                x_sentence_bert, x_sentence_token_embedding = self.sentence_embedding_bert(recog_hidden, gt_texts)

                if self.merge_type == 'Sum':
                    segment_emb = recog_hidden + x_sentence_token_embedding
                    segment_emb = self.layernorm(segment_emb)
                elif self.merge_type == 'Gate':
                    b_, n_, l_, d_ = recog_hidden.size()
                    recog_hidden = recog_hidden.view(-1, d_).contiguous()
                    x_sentence_token_embedding_tmp = x_sentence_token_embedding.view(-1,
                                                                                     x_sentence_token_embedding.size(
                                                                                         -1))
                    gate = torch.sigmoid(self.xx(recog_hidden) + self.yy(x_sentence_token_embedding_tmp))
                    new_info = torch.tanh(self.xx_1(recog_hidden) + self.yy_1(x_sentence_token_embedding_tmp))
                    segment_emb = recog_hidden + (gate * new_info)
                    segment_emb = segment_emb.view(b_, n_, l_, d_).contiguous()
                elif self.merge_type == 'GateNew':
                    b_, n_, l_, d_ = recog_hidden.size()
                    recog_hidden = recog_hidden.view(-1, d_).contiguous()
                    x_sentence_token_embedding_tmp = x_sentence_token_embedding.view(-1,
                                                                                     x_sentence_token_embedding.size(
                                                                                         -1))
                    gate = torch.sigmoid(self.xx(recog_hidden) + self.yy(x_sentence_token_embedding_tmp))
                    new_info = self.relu(self.yy_1(x_sentence_token_embedding_tmp))
                    segment_emb = recog_hidden + (gate * new_info)
                    segment_emb = segment_emb.view(b_, n_, l_, d_).contiguous()
                elif self.merge_type == 'SumNew':
                    b_, n_, l_, d_ = recog_hidden.size()
                    recog_hidden = recog_hidden.view(-1, d_).contiguous()
                    x_sentence_token_embedding_tmp = x_sentence_token_embedding.view(-1,
                                                                                     x_sentence_token_embedding.size(
                                                                                         -1))
                    segment_emb = recog_hidden + self.relu(self.yy(x_sentence_token_embedding_tmp))
                    segment_emb = segment_emb.view(b_, n_, l_, d_).contiguous()
                    segment_emb = self.layernorm(segment_emb)
                else:
                    raise "Unknown merge type {}".format(self.merge_type)

                ret_sentence_token_embedding = segment_emb

            segment_emb = self.sentence_embedding(segment_emb, char_nums)

        segment_emb = self.dropout(segment_emb)

        return segment_emb, ret_sentence_token_embedding
