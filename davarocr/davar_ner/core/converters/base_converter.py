"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    base_converter.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
from abc import ABCMeta, abstractmethod


class BaseConverter(metaclass=ABCMeta):
    """ Base converter, Convert between text, index and tensor for NER pipeline.
    """
    @abstractmethod
    def convert_text2id(self, results):
        """ Convert token to ids.

        Args:
            results (dict): A dict must containing the token key:
                    - tokens (list]): Tokens list.
        Returns:
            dict: corresponding ids
        """
        pass

    @abstractmethod
    def convert_pred2entities(self, preds, masks, **kwargs):
        """ Gets entities from preds.

        Args:
            preds (list): Sequence of preds.
            masks (Tensor): The valid part is 1 and the invalid part is 0.
        Returns:
            list: List of entities.
        """
        pass

    @abstractmethod
    def convert_entity2label(self, labels):
        """ Convert labeled entities to ids.

        Args:
            labels (list): eg:['B-PER', 'I-PER']
        Returns:
            dict: corresponding labels
        """
        pass
