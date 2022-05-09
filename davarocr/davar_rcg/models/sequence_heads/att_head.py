"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    att_head.py
# Abstract       :    Implementations of the Attn prediction layer, loss calculation and result converter

# Current Version:    1.0.0
# Date           :    2021-05-01
# Thanks to      :    We borrow the released code from http://gitbug.com/clovaai/deep-text-recognition-benchmark
                      for the AttentionCell.
##################################################################################################
"""
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from mmdet.models.builder import HEADS
from mmdet.models.builder import build_loss
from mmcv.runner import load_checkpoint

from davarocr.davar_common.core.builder import build_converter


@HEADS.register_module()
class AttentionHead(nn.Module):
    """Attention Recognition Head proposed in Ref [1] and Ref [2]

       Ref [1]: Robust Scene Text Recognition with Automatic Rectification. CVPR-2016
       Ref [2]: What Is Wrong With Scene Text Recognition Model Comparisons Dataset and Model Analysis ICCV-2019
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_max_length=25,
                 loss_att=dict(
                    type='CELoss',
                    ignore_index=-100,
                    weight=1.0,
                    reduction='mean'),
                 converter=dict(
                    type='AttnLabelConverter',
                    character='0123456789abcdefghijklmnopqrstuvwxyz',)
                 ):
        """
        Args:
            input_size (int): input size
            hidden_size (int): hidden  state size
            batch_max_length (int): batch max text length
            loss_att (dict): loss function parameter
            converter (dict): converter parameter
        """

        super(AttentionHead, self).__init__()
        self.hidden_size = hidden_size
        self.batch_max_length = batch_max_length

        # build the loss
        self.loss_att = build_loss(loss_att)

        # build the converter
        self.converter = build_converter(converter)
        self.num_classes = len(self.converter.character)
        self.bos = self.converter.bos

        attention_cell_num = self.num_classes

        self.attention_cell = AttentionCell(input_size,
                                            hidden_size,
                                            attention_cell_num)
        self.generator = nn.Linear(hidden_size, self.num_classes)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("AttentionHead:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, param in self.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        """

        Args:
            input_char (tensor): input tensor
            onehot_dim (int): one hot dim

        Returns:
            Torch.tensor: transformed one hot vector corresponding to the input

        """
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)

        # initialize the one hot tensor
        one_hot = torch.cuda.FloatTensor(batch_size,
                                         onehot_dim,
                                         device=input_char.device).zero_()
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def _char_to_embedding(self, input_char):
        """
        Args:
            input_char (tensor): input vector

        Returns:
            Torch.tensor: transformed embedding vector corresponding to the input

        """
        embedding = self.vecs[input_char, :]
        return embedding

    def forward(self, batch_H, target,
                is_train=True,
                return_hidden=False):
        """
        Args:
            batch_H (Torch.Tensor): contextual_featurec H = hidden state of encoder.
                                    [batch_size x num_steps x num_classes]
            target (Torch.Tensor): the text-index of each image.[batch_size x (max_length+1)].
                                    +1 for [GO] token. text[:, 0] = [GO].
            is_train (bool): whether is training state or test state
            return_hidden (bool): whether to return hidden state

        Returns:
            Torch.Tensor: probability distribution at each step [batch_size x num_steps x num_classes]

        """
        if target is not None:
            gt_label, _ = target
        batch_size = batch_H.size(0)
        num_steps = self.batch_max_length + 1  # +1 for [s] at end of sentence. # 31

        output_hiddens = torch.cuda.FloatTensor(batch_size,
                                                num_steps,
                                                self.hidden_size).fill_(0)
        hidden = (torch.cuda.FloatTensor(batch_size, self.hidden_size,
                                         device=batch_H.device).fill_(0),
                  torch.cuda.FloatTensor(batch_size, self.hidden_size,
                                         device=batch_H.device).fill_(0))
        if is_train:
            for i in range(num_steps):
                # The vector corresponding to the i-th text in one batch
                char_onehots = self._char_to_onehot(gt_label[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)

            probs = self.generator(output_hiddens)

        else:
            targets = torch.cuda.LongTensor(batch_size,
                                            device=batch_H.device).fill_(self.bos)  # [GO] token
            probs = torch.cuda.FloatTensor(batch_size,
                                           num_steps,
                                           self.num_classes,
                                           device=batch_H.device).fill_(0)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)

                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        if return_hidden:
            return probs, output_hiddens

        return probs  # batch_size x num_steps x num_classes

    def convert(self, text):
        """
        Args:
            text (str): text format information

        Returns:
            Torch.Tensor: label information after converter

        """
        return self.converter(text)

    def loss(self, pred, target):
        """

        Args:
            pred (Torch.Tensor): model prediction
            target (Torch.Tensor): label information

        Returns:
            Torch.Tensor: model training loss

        """
        loss = dict()
        gt_label, _ = target
        gt_label = gt_label[:, 1:]  # without [GO] Symbol shape

        loss_att = self.loss_att(pred.view(-1, pred.shape[-1]),
                                 gt_label.contiguous().view(-1))
        loss['loss_att'] = loss_att

        return loss

    def get_target(self, gt_texts):
        """
        Args:
            gt_texts (str): text format label information

        Returns:
            Torch.Tensor: vector transformed by text label
        Returns:
            Torch.Tensor: vector transformed by text length

        """
        if gt_texts is not None:
            text, length = self.converter.encode(gt_texts, self.batch_max_length)
            return text, length

        return None, None

    def get_pred_text(self, preds, batch_max_length):
        """

        Args:
            preds (tensor): model output feature
            batch_max_length (int): max output text length

        Returns:
            list(str): true text format prediction transformed by the converter

        """
        batch_size = preds.size(0)
        length_for_pred = torch.cuda.IntTensor([batch_max_length] * batch_size)

        preds = preds[:, :batch_max_length, :]
        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, length_for_pred)

        # transfer the model prediction to text
        preds_str = [pred_str[:pred_str.find('[s]')] for pred_str in preds_str]
        return preds_str


class AttentionCell(nn.Module):
    """ Attention Cell Structure """
    def __init__(self, input_size, hidden_size, num_embeddings):

        """
        Args:
            input_size (int): input channel
            hidden_size (int): hidden state num
            num_embeddings (int): embedding layers
        """
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)  # 512->256
        self.h2h = nn.Linear(hidden_size, hidden_size)             # either i2i or h2h should have bias 256->256
        self.score = nn.Linear(hidden_size, 1, bias=False)         # 256->1
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)  # 512+1w->256
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_h, char_onehots):
        """
        Args:
            prev_hidden (Torch.Tensor): previous layer's hidden state
            batch_h (Torch.Tensor): sequential input feature
            char_onehots (Torch.Tensor): one hot vector

        Returns:
            Torch.Tensor: current hidden state
        Returns:
            Torch.Tensor: attention weight

        """
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_h_proj = self.i2h(batch_h)

        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)

        emphasis = self.score(torch.tanh(batch_h_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(emphasis, dim=1)

        context = torch.bmm(alpha.permute(0, 2, 1), batch_h).squeeze(1)     # batch_size x num_channel
        
        concat_context = torch.cat([context, char_onehots], 1)              # batch_size x (num_channel + num_embedding)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha
