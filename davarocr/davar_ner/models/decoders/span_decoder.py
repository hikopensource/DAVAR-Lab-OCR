"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    span_decoder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models.builder import build_loss
from davarocr.davar_nlp_common.models import BaseDecoder
from davarocr.davar_nlp_common.models.builder import DECODERS
from .layers import PoolerStartLogits, PoolerEndLogits


@DECODERS.register_module()
class SpanDecoder(BaseDecoder):
    """ Span decoder class for NER.
    """
    def __init__(self,
                 hidden_dropout_prob=0.1,
                 hidden_size=768,
                 soft_label=True,
                 loss=dict(type='StandardCrossEntropyLoss',ignore_index=-100),
                 **kwargs):
        """
        Args:
            hidden_dropout_prob (float): The dropout probability of hidden layer.
            hidden_size (int): Hidden layer output layer channels.
            soft_label (bool): whether use soft label.
            loss (dict): the loss function.
        """
        super().__init__(**kwargs)
        self.num_labels = self.label_converter.num_labels
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.soft_label = soft_label

        self.start_fc = PoolerStartLogits(hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(hidden_size + 1, self.num_labels)
        self.criterion = build_loss(loss)

    def extract_feat(self, outputs, **kwargs):
        """ extract feat on training stage.
        """
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        start_positions = kwargs['start_positions']
        input_ids = kwargs['input_ids']

        if self.soft_label:
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
            label_logits.zero_()
            label_logits = label_logits.to(input_ids.device)
            label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
        else:
            label_logits = start_positions.unsqueeze(2).float()

        end_logits = self.end_fc(sequence_output, label_logits)
        return start_logits, end_logits

    def forward(self, outputs, **kwargs):
        return self.extract_feat(outputs, **kwargs)

    def loss(self, logits, **kwargs):
        """ Caculates the loss between the start/end logits and gt.

        Args:
            logits(tensor): the network output.
            **kwargs(dict): the ground trunth contains the `start_positions`, `end_positions`
        and `attentino_masks` keys.
        
        Returns:
            dict: loss
        """
        start_logits, end_logits = logits
        attention_mask = kwargs['attention_masks']
        start_positions = kwargs['start_positions']
        end_positions = kwargs['end_positions']

        start_logits = start_logits.view(-1, self.num_labels)
        end_logits = end_logits.view(-1, self.num_labels)
        active_loss = attention_mask.view(-1) == 1
        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]

        active_start_labels = start_positions.view(-1)[active_loss]
        active_end_labels = end_positions.view(-1)[active_loss]
        start_loss = self.criterion(active_start_logits, active_start_labels)
        end_loss = self.criterion(active_end_logits, active_end_labels)
        return {'start_loss':start_loss, 'end_loss': end_loss}

    def decode(self, outputs, **kwargs):
        """ span decode
        """
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)

        label_logits = F.softmax(start_logits, -1)
        if not self.soft_label:
            label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        res = self._extract_item(start_logits, end_logits, kwargs['attention_masks'])
        return res

    def _extract_item(self, start_logits, end_logits, attention_masks):
        """ get entities from preds
        """
        res = []
        batch_size = start_logits.shape[0]
        for batch in range(batch_size):
            attention_mask = attention_masks.cpu().numpy()[batch]
            start_pred = torch.argmax(start_logits, -1).cpu().numpy()[batch]
            end_pred = torch.argmax(end_logits, -1).cpu().numpy()[batch]
            entity = []
            for i, s_l in enumerate(start_pred):
                if s_l == 0 or attention_mask[i] == 0:
                    continue
                for j, e_l in enumerate(end_pred[i:]):
                    if s_l == e_l:
                        entity.append((s_l, i, i+j+1))
                        break
            res.append(entity)
        return res
