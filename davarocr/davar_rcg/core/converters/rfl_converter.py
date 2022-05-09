"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    rfl_converter.py
# Abstract       :    Implementations of text-label and text-index of rf-learning counting converter

# Current Version:    1.0.0
# Date           :    2021-04-30
##################################################################################################
"""
import os.path as osp
import json

import torch
from mmcv.utils import print_log

from davarocr.davar_common.core.builder import CONVERTERS


@CONVERTERS.register_module()
class RFLLabelConverter:
    """ Convert between text-label and text-index in Ref [1]

    Ref [1]: Reciprocal Feature Learning via Explicit and Implicit Tasks in Scene Text Recognition. ICDAR-2021.
    """

    def __init__(self, character,
                 use_cha_eos=True,
                 cates=10000):
        """
            Convert between text-label and text-index
        Args:
            character (str): set of the possible recognition characters dictionary.
            use_cha_eos (bool): use the character define
            cates (int): category number of the character
        """

        self.bos = 0
        self.eos = 1

        # character dictionary is file format
        if osp.isfile(character):
            with open(character, 'r', encoding='utf-8') as character_file:
                # character dictionary is json file
                if character.endswith('.json'):
                    print_log("loading user predefined recognition dictionary from json file: "
                              "%s to build the CNT converter !!!" % character)

                    character = json.load(character_file)
                    assert 'char2index' in character
                    self.character = ['' for _ in range(cates)]
                    if use_cha_eos:
                        self.dict = character['char2index']
                        self.bos = self.dict['BOS']
                        self.eos = self.dict['EOS']
                        for key, value in self.dict.items():
                            self.character[value] = key
                    else:
                        self.dict = dict()
                        for key, value in character['char2index'].items():
                            self.dict[key] = value + 2
                            self.character[value + 2] = key

                    self.character[self.bos] = '[GO]'
                    self.character[self.eos] = '[s]'
                    self.dict['[GO]'] = self.character.index('[GO]')
                    self.dict['[s]'] = self.character.index('[s]')

                # character dictionary is txt file
                elif character.endswith('.txt'):
                    print_log("loading user predefined recognition dictionary from txt file: "
                              "%s to build the CNT converter !!!" % character)

                    character = character_file.readline().strip()

                    # [GO] for the start token of the attention decoder.
                    # [s] for end-of-sentence token.
                    list_token = ['[GO]', '[s]']

                    list_character = list(character)
                    self.character = list_token + list_character
                    self.dict = {}
                    for i, char in enumerate(self.character):
                        self.dict[char] = i
                else:
                    raise Exception("dictionary file type is not support !!!")
        elif ".json" in character or ".txt" in character:
            # character file does not exist, raise the error
            raise FileNotFoundError("The recognition character file is not existing")
        else:
            raise Exception("dictionary file only support the txt and json file !!!")

        print("recognition dictionary %s \t" % str(self.dict).encode(encoding="utf-8").decode(encoding="utf-8"))

    def encode(self, text):
        """
            convert text-label into text-index.
        Args:
            text (list): text labels of each image. [batch_size]

        Returns:
            Torch.Tensor : the training target. [batch_size x (character_num)].
                   text[:, :] is the character occurrence.

        """

        batch_one_hot = torch.cuda.FloatTensor(len(text), len(self.character)).fill_(0.0)
        for i, item in enumerate(batch_one_hot):
            for char_ in text[i]:
                try:
                    item[self.dict[char_]] += 1
                except Exception as DictionaryError:
                    raise KeyError from DictionaryError

        return batch_one_hot

    def decode(self, pred):
        """
            convert text-index into text-label.
        Args:
            pred (Torch.tensor): the output of the model

        Returns:
            list(int): model pred text length
        """

        pred_length = []
        for lens in pred:
            # transfer the model prediction to text length
            length = round(torch.sum(lens).item())
            pred_length.append(length)

        return pred_length
