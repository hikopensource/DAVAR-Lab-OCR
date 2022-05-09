"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    fc_decoder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import build_loss
from davarocr.davar_nlp_common.models import BaseDecoder
from davarocr.davar_nlp_common.models.builder import DECODERS


@DECODERS.register_module()
class FCDecoder(BaseDecoder):
    """ FC decoder class for Ner.
    """
    def __init__(self,
                 hidden_dropout_prob=0.1,
                 hidden_size=768,
                 loss=dict(type='StandardCrossEntropyLoss',ignore_index=-100),
                 **kwargs):
        """
        Args:
            hidden_dropout_prob (float): The dropout probability of hidden layer.
            hidden_size (int): Hidden layer output layer channels.
            loss (dict): loss function.
        """
        super().__init__(**kwargs)
        self.num_labels = self.label_converter.num_labels
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.criterion = build_loss(loss)

    def forward(self, outputs, **kwargs):
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
    
    def loss(self, logits, **kwargs):
        """ Caculates the loss between logits and gt.

        Args:
            logits(tensor): the network output.
            **kwargs(dict): the ground trunth contains the `labels` and `attentino_masks` keys.
        
        Returns:
            dict: loss
        """
        labels = kwargs['labels']
        attention_masks = kwargs['attention_masks']

        # Only keep active parts of the loss
        if attention_masks is not None:
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        else:
            loss = self.criterion(
                logits.view(-1, self.num_labels), labels.view(-1))
        return {'loss_cls': loss}

    def decode(self, outputs, **kwargs):
        """ Decode method
        """
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        softmax = F.softmax(logits, dim=2)
        preds = softmax.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        res = []
        masks = kwargs['attention_masks']
        masks = masks.detach().cpu().numpy()
        for index, pred in enumerate(preds):
            results = (masks[index] * np.array(pred)).tolist()
            res.append(results)
        return res
