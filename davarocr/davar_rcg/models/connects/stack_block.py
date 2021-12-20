"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    stack_block.py
# Abstract       :    Implementations of the stack structure related information

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import logging

import torch.nn as nn
import torch.nn.init as init

from mmcv.runner import load_checkpoint

from davarocr.davar_common.models.builder import CONNECTS
from davarocr.davar_common.models.builder import build_connect


@CONNECTS.register_module()
class CascadeRNN(nn.Module):
    """ Cascade Recurrent Structure"""
    def __init__(self, rnn_modules,
                 repeat=None,
                 mode=1,
                 ):
        """
        Args:
            rnn_modules (nn.Module): stack structure(RNN|LSTM)
            repeat (int): repeat times
            mode (int): use mode,  0 - without mask, output dim [batch_size x C x height x width]
                                   1 - with mask, output dim [batch_size x T x output_size]
        """
        super(CascadeRNN, self).__init__()
        if repeat:
            assert len(rnn_modules) == 1 or len(rnn_modules) == repeat
            modules = []
            for _ in range(repeat):
                # build the rnn_modules
                modules.append(build_connect(rnn_modules[0]))
        else:
            # build the rnn_module
            modules = [build_connect(rnn_module) for rnn_module in rnn_modules]

        self.SequenceModeling = nn.Sequential(*modules)
        self.mode = mode

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("CascadeRNN:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, param in self.SequenceModeling.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs (torch.Tensor): input feature
            mask (torch.Tensor): input mask, according to the structure and specify the corresponding mask

        different mode with different input feature:
            mode: 1 (default)
            input : visual feature      [batch_size x T x input_size]
            output : contextual feature [batch_size x T x output_size]
            mode: 0
            input : visual feature      [batch_size x C x height x width]
            output : contextual feature [batch_size x C x height x width]

        Returns:
            torch.Tensor: output feature of the stack structure
        """

        if self.mode:
            output = self.SequenceModeling(inputs)
            if mask is not None:
                output = inputs
                for module in self.SequenceModeling:
                    output = module(output, mask=mask)
            return output

        output = self.SequenceModeling(inputs.contiguous().view(inputs.size(0),
                                                                inputs.size(1), -1).permute(0, 2, 1))
        return output.permute(0, 2, 1).unsqueeze(2)


@CONNECTS.register_module()
class CascadeCNN(nn.Module):
    """ Cascade Convolution Structure"""
    def __init__(self,
                 cnn_modules,
                 repeat=None,
                 mode=0,
                 ):
        """
        Args:
            cnn_modules (nn.Module): stack structure(CNN)
            repeat (int): repeat times
            mode (int): use mode,  0 - without mask, output dim [batch_size x C x height x width]
                                   1 - with mask, output dim [batch_size x T x output_size]
        """
        super(CascadeCNN, self).__init__()

        if repeat:
            assert len(cnn_modules) == 1 or len(cnn_modules) == repeat
            modules = []
            for _ in range(repeat):
                modules.append(build_connect(cnn_modules[0]))
        else:
            modules = [build_connect(cnn_module)
                       for cnn_module in cnn_modules]

        self.SequenceModeling = nn.Sequential(*modules)
        self.mode = mode

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("CascadeCNN:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, param in self.SequenceModeling.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    if 'bn' in name:
                        init.constant_(param, 1.0)
                    else:
                        init.kaiming_normal_(param)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input_feature):
        """

        Args:
            (torch.Tensor): input feature

        different mode with different input feature:
            mode-0 (default)
            input_feature :  visual feature     [batch_size x C x height x width]
            output : contextual feature         [batch_size x C x height x width]
            mode: 1
            input_feature : visual feature      [batch_size x T x input_size]
            output : contextual feature         [batch_size x T x output_size]
        Returns:
            torch.Tensor: output feature of the stack structure

        """

        if self.mode:
            output = self.SequenceModeling(input_feature.unsqueeze(-1).permute(0, 2, 1, 3))
            return output.permute(0, 2, 1, 3).squeeze(3)

        output = self.SequenceModeling(input_feature)
        return output
