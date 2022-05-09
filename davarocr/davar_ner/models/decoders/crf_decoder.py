"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    crf_decoder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import torch.nn as nn
from .layers import CRF
from davarocr.davar_nlp_common.models import BaseDecoder
from davarocr.davar_nlp_common.models.builder import DECODERS


@DECODERS.register_module()
class CRFDecoder(BaseDecoder):
    """ CRF decoder class for Ner.
    """
    def __init__(self,
                 hidden_dropout_prob=0.1,
                 hidden_size=768,
                 **kwargs):
        """
        Args:
            hidden_dropout_prob (float): The dropout probability of hidden layer.
            hidden_size (int): Hidden layer output layer channels.
        """
        super().__init__(**kwargs)
        self.num_labels = self.label_converter.num_labels
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(self, outputs, **kwargs):
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

    def loss(self, logits, **kwargs):
        loss = self.crf(logits, kwargs['labels'].squeeze(1), kwargs['attention_masks'].squeeze(1))
        return {'loss_crf': -loss}

    def decode(self, outputs, **kwargs):
        """ Crf decode process
        """
        logits = self.forward(outputs, **kwargs)
        preds = self.crf.decode(logits, kwargs['attention_masks'].squeeze(1))
        preds = preds.squeeze(0).cpu().numpy()
        preds = preds.tolist()
        return preds
