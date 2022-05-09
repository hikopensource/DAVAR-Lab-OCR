"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    encoder_decoder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-08-02
##################################################################################################
"""
from abc import ABCMeta
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.distributed as dist
from ..builder import build_decoder, build_encoder


class EncoderDecoder(nn.Module, metaclass=ABCMeta):
    """Encoder decoder framework for nlp methods."""

    def __init__(self,
                 encoder,
                 decoder,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def extract_feat(self, imgs):
        """Extract features from images."""
        raise NotImplementedError(
            'Extract feature module is not implemented yet.')

    def forward(self, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    def forward_train(self, input_ids, attention_masks=None, token_type_ids=None, **kwargs):
        """ calls on training stage.

        Args:
            input_ids(tensor): the input ids corresponding to the text.
            attention_masks(tensor): attention_mask.
            token_type_ids(tensor): token type ids.
        Returns:
            dict: the network's loss.
        """
        encode_out = self.encoder(input_ids=input_ids,
                                  attention_masks=attention_masks,
                                  token_type_ids=token_type_ids,
                                  **kwargs)
        loss = self.decoder.forward_train(encode_out,
                              input_ids=input_ids,
                              attention_masks=attention_masks,
                              token_type_ids=token_type_ids,
                              **kwargs)
        return loss

    def forward_test(self, input_ids, attention_masks=None, token_type_ids=None, **kwargs):
        """ Calls either :func:`aug_test` or :func:`simple_test` depending
        on whether ``input_ids`` is ``list``.
        """
        if isinstance(input_ids, list):
            assert len(input_ids) > 0
            assert input_ids[0].size(0) == 1, ('aug test does not support '
                                          f'inference with batch size '
                                          f'{input_ids[0].size(0)}')
            return self.aug_test(input_ids, attention_masks, token_type_ids, **kwargs)
        return self.simple_test(input_ids, attention_masks, token_type_ids, **kwargs)

    def aug_test(self, input_ids, attention_masks=None, token_type_ids=None, **kwargs):
        """ Augmentation test.
        """
        raise NotImplementedError

    def simple_test(self, input_ids, attention_masks=None, token_type_ids=None, **kwargs):
        """ Simple test.
        """
        encode_out = self.encoder(input_ids=input_ids,
                                  attention_masks=attention_masks,
                                  token_type_ids=token_type_ids,
                                  **kwargs)
        preds = self.decoder.forward_test(encode_out,
                              input_ids=input_ids,
                              attention_masks=attention_masks,
                              token_type_ids=token_type_ids,
                              **kwargs)
        return preds

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw outputs of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer update, which are done by an optimizer
        hook. Note that in some complicated cases or models (e.g. GAN),
        the whole process (including the back propagation and optimizer update)
        is also defined by this method.

        Args:
            data (dict): The outputs of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which is a
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size used for
                averaging the logs (Note: for the
                DDP model, num_samples refers to the batch size for each GPU).
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['input_ids']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but is
        used during val epochs. Note that the evaluation after training epochs
        is not implemented by this method, but by an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['input_ids']))

        return outputs
