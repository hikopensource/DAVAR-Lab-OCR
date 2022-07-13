"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    base.py
# Abstract       :    Implementations of the Recognition Base Class Structure

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
import logging
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import torch.distributed as dist


class BaseRecognizor(nn.Module):
    """Base class for recognizors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        """
        super().__init__()
        self.fp16_enabled = False

    @property
    def with_transformation(self):
        """
        Returns:
            bool: Determine the model is with the transformation or not

        """
        return hasattr(self, 'transformation') and self.transformation is not None

    @property
    def with_converter(self):
        """

        Returns:
            bool: Determine the model is with the converter or not

        """

        return hasattr(self, 'converter') and self.converter is not None

    @abstractmethod
    def forward_train(self, imgs,
                      gt_texts,
                      img_meta=None,
                      **kwargs):
        """
        Args:
            imgs (Torch.tensor): input image
            gt_texts (Torch.tensor): label information
            img_meta (None): meta information
            **kwargs (None): backup parameter

        Returns:

        """
        pass

    @abstractmethod
    def simple_test(self, imgs,
                    gt_texts=None,
                    img_meta=None,
                    **kwargs):
        """
        Args:
            imgs (Torch.tensor): input image
            gt_texts (Torch.tensor): label information
            img_meta (None): meta information
            **kwargs (None): backup parameter

        """
        pass
    
    @abstractmethod
    def aug_test(self, imgs, gt_texts, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        """
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs,
                     gt_texts=None,
                     img_meta=None,
                     **kwargs):
        """
        Args:
            imgs (Torch.tensor): input image
            gt_texts (Torch.tensor): label information
            img_meta (None): meta information
            **kwargs (None): backup parameter

        Returns:
            dict: result of the model inference

        """

        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]

        num_augs = len(imgs)
        if num_augs == 1:
            # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
            return self.simple_test(imgs[0],
                                    gt_texts[0]
                                    if gt_texts is not None else None, **kwargs)
        else:
            return self.aug_test(imgs, gt_texts if gt_texts is not None else None, **kwargs)

    def forward(self, img, gt_text=None,
                img_meta=None, return_loss=True,
                **kwargs):
        """
        Args:
            img (Torch.tensor): input image
            gt_text (Torch.tensor): label information
            img_meta (None): meta information
            return_loss (bool): whether to return loss
            **kwargs (None): backup parameter

        Returns:
            dict: result of the model forward or inference

        """
        if return_loss:
            return self.forward_train(img, gt_text, **kwargs)

        return self.forward_test(img, gt_texts=gt_text, **kwargs)

    def _parse_losses(self, losses):
        """
        Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Torch.tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
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
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img']))

        return outputs

    def val_step(self, data, optimizer):
        """
           The iteration step during validation.

            This method shares the same signature as :func:`train_step`, but used
            during val epochs. Note that the evaluation after training epochs is
            not implemented with this method, but an evaluation hook.

        Args:
            data (Torch.tensor): validation image
            optimizer (optimizer): model optimizer

        Returns:
            dict: validation result of the training process

        """

        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img']))

        return outputs
