"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    base_decoder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-08-02
##################################################################################################
"""
import torch.nn as nn
from mmcv.cnn import uniform_init, xavier_init

from ..builder import DECODERS
from davarocr.davar_common.core import build_converter


@DECODERS.register_module()
class BaseDecoder(nn.Module):
    """ Base Decoder. forward_train will be called in training stage 
    and returns the loss, forward_test will be called in test stage.
    """
    def __init__(self,
                 label_converter):
        """
        Args:
            label_converter(dict): label converter
        """
        super().__init__()
        self.label_converter = build_converter(label_converter)
        self.init_weights()

    def forward(self, outputs, **kwargs):
        raise NotImplementedError(
            'forward module is not implemented yet.')
        
    def loss(self, **kwargs):
        """Compute losses of the head."""
        raise NotImplementedError(
            'loss is not implemented yet.')

    def forward_train(self, outputs, **kwargs):
        logits = self.forward(outputs, **kwargs)
        loss = self.loss(logits, **kwargs)
        return loss

    def decode(self, outputs, **kwargs):
        return self.forward(outputs, **kwargs)

    def forward_test(self, outputs, **kwargs):
        """Decode pred results on test stage and convert to entities.
        """
        preds = self.decode(outputs, **kwargs)
        attention_masks = kwargs.get('attention_masks', None)
        pred_entities = self.label_converter.convert_pred2entities(
            preds, attention_masks,**kwargs)
        return pred_entities

    def init_weights(self):
        """Init weights for decoder
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                xavier_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                uniform_init(module)
