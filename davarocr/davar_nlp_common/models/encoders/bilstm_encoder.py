"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    bilstm_encoder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-09-01
##################################################################################################
"""
import os
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from ..builder import ENCODERS


@ENCODERS.register_module()
class BiLSTMEncoder(nn.Module):
    """ BiLSTM encoder
    """
    def __init__(self,
                 vocab_path,
                 emb_size=128,
                 num_lstm=1,
                 hidden_size=256):
        """
        Args:
            vocab_path(str):the model's path that include vocab.txt
            emb_size (int) embedding dimension
            hidden_size (int): output dimension
        """
        super().__init__()

        assert os.path.exists(vocab_path), "{} is not exist!".format(vocab_path)
        with open(vocab_path, encoding='utf-8') as file:
            data = file.read().split('\n')
        vocab_size = len(data)
        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.first_bilstm = nn.LSTM(emb_size, hidden_size//2,
                              batch_first=True,
                              bidirectional=True)
        self.num_lstm = num_lstm
        self.bilstm_layers = []
        for i in range(num_lstm-1):
            layer_name = 'layer%s'%(i+1)
            lstm_layer = nn.LSTM(hidden_size, hidden_size//2,
                              batch_first=True,
                              bidirectional=True)
            self.add_module(layer_name, lstm_layer)
            self.bilstm_layers.append(layer_name)
        

    def forward(self, **results):
        input_ids = results['input_ids']
        lengths = [len(l) for l in input_ids]
        emb = self.embedding(input_ids)  # shape[B, L, emb_size]
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.first_bilstm(packed)
        if self.bilstm_layers:
            for layer_name in self.bilstm_layers:
                bilstm_layer = getattr(self, layer_name)
                rnn_out, _ = bilstm_layer(rnn_out)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True) #shape[B,L,hidden_size]
        return [rnn_out]
