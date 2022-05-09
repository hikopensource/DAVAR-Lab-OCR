"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    transformers_encoder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from ..builder import ENCODERS


@ENCODERS.register_module()
class TransformersEncoder(nn.Module):
    """ Transformers encoder
    """
    def __init__(self,
                 model_name_or_path,
                 with_token_type_ids=True,
                 use_last_hidden_states=0,
                 lstm=False,
                 hidden_size=768):
        """
        Args:
            model_name_or_path(str): the pretrained model's name or path.
            with_token_type_ids(bool): whether use token_type_ids as model's input.
            use_last_hidden_states (int): use the concatenation of last k hidden states to get the hidden state.
            lstm (bool): whether use bilstm on hidden state.
            hidden_size (int): the size of the hidden state.
        """
        super().__init__()
        self.transformers_model = AutoModel.from_pretrained(model_name_or_path)
        self.with_token_type_ids = with_token_type_ids
        self.use_last_hidden_states = use_last_hidden_states
        self.lstm = lstm
        if lstm:
            self.bilstm = nn.LSTM(hidden_size, hidden_size//2,
                              batch_first=True,
                              bidirectional=True)

    def forward(self, **results):
        device = next(self.transformers_model.parameters()).device
        input_ids = results['input_ids'].to(device)
        attention_masks = results['attention_masks'].to(device)
        token_type_ids = results['token_type_ids'].to(device)
        if self.with_token_type_ids:
            outputs = self.transformers_model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids)
        else:
            outputs = self.transformers_model(
                input_ids=input_ids,
                attention_mask=attention_masks
                )
        if self.use_last_hidden_states > 0:
            pooled_output = outputs[1]
            hidden_states = outputs[2][-self.use_last_hidden_states:]
            last_hidden_states = outputs[0]
            cat = [pooled_output]
            for hidden_state in hidden_states:
                cat.append(hidden_state[:,0])
            cat = torch.cat(cat,1)
            outputs = (last_hidden_states,cat)
        if self.lstm:
            rnn_out, _ = self.bilstm(outputs[0])
            return (rnn_out, outputs[1])
        return outputs
