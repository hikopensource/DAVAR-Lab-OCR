"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    LSTM.py
# Abstract       :    Implementations of Bidirectional LSTM

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import torch.nn as nn

from davarocr.davar_common.models.builder import CONNECTS


@CONNECTS.register_module()
class BidirectionalLSTM(nn.Module):
    """
    input : visual feature      [batch_size x T x input_size]
    output : contextual feature [batch_size x T x output_size]

    Usage sample:
    sequence_module=dict(
            type='CascadeRNN',
            mode=1,
            repeat=2,
            rnn_modules=[
            dict(
                type='BidirectionalLSTM',
                input_size=512,
                hidden_size=256,
                output_size=512,
            ),
            dict(
                type='BidirectionalLSTM',
                input_size=512,
                hidden_size=256,
                output_size=512,
        ),
    ]),
    """

    def __init__(self, input_size,
                 hidden_size,
                 output_size=None,
                 num_layers=1,
                 dropout=0,
                 bidirectional=False,
                 batch_first=True,
                 with_linear=False):
        """

        Args:
            input_size (int): input feature dim
            hidden_size (int): hidden state dim
            output_size (int): output feature dim
            num_layers (int): layers of the LSTM
            dropout (float): probability of the dropout
            bidirectional (bool): bidirectional interface, default False
            batch_first (bool): feature format ''(batch, seq, feature)''
            with_linear (bool): whether to combine linear layer with LSTM
        """

        super(BidirectionalLSTM, self).__init__()
        self.with_linear = with_linear
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=batch_first)

        # text recognition the specified structure LSTM with linear
        if self.with_linear:
            self.linear = nn.Linear(hidden_size * 2, output_size)

    def init_weights(self, pretrained=None):
        """
        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """

    def forward(self, input_feature):
        """

        Args:
            input_feature (Torch.Tensor): visual feature [batch_size x T x input_size]

        Returns:
            Torch.Tensor: LSTM output contextual feature [batch_size x T x output_size]

        """

        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input_feature)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        if self.with_linear:
            output = self.linear(recurrent)     # batch_size x T x output_size
            return output
        return recurrent
