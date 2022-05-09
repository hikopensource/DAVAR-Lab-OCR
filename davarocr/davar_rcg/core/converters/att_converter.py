"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    att_converter.py
# Abstract       :    Implementations of text-label and text-index of Attn

# Current Version:    1.0.0
# Date           :    2021-04-30
# Thanks to      :    We borrow the released code from http://gitbug.com/clovaai/deep-text-recognition-benchmark
                      for the AttnLabelConverter.
##################################################################################################
"""
import os.path as osp
import json

import torch
import numpy as np
from mmcv.utils import print_log

from davarocr.davar_common.core.builder import CONVERTERS


@CONVERTERS.register_module()
class AttnLabelConverter:
    """Convert between text-label and text-index"""
    def __init__(self, character,
                 use_cha_eos=True,
                 cates=10000,
                 with_unknown=False):
        """
            Convert between text-label and text-index
        Args:
            character (str): set of the possible recognition characters dictionary.
            use_cha_eos (bool): use the character define
            cates (int): category number of the character
            with_unknown (bool): whether to encode the characters which are out of the dictionary to ['[UNK]']
        """

        self.bos = 0
        self.eos = 1

        self.with_unknown = with_unknown
        # character dictionary is file format
        if osp.isfile(character):
            with open(character, 'r', encoding='utf-8') as character_file:
                # character dictionary is json file
                if character.endswith('.json'):
                    print_log("loading user predefined recognition dictionary from json file: "
                              "%s to build the ATT converter !!!" % character)
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
                              "%s to build the ATT converter !!!" % character)
                    character = character_file.readline().strip()

                    # [GO] for the start token of the attention decoder.
                    # [s] for end-of-sentence token.
                    list_token = ['[GO]', '[s]']
                    if self.with_unknown:
                        unk_token = ['[UNK]']
                    else:
                        unk_token = list()
                    list_character = list(character)
                    self.character = list_token + list_character + unk_token
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

    def encode(self, text, batch_max_length=25):
        """
            convert text-label into text-index.
        Args:
            text (list): text labels of each image. [batch_size]
            batch_max_length (tensor): max length of text label in the batch. 25 by default

        Returns:
            Torch.Tensor : the input of attention decoder. [batch_size x (character_num)].
                text[:, 0] is text length and text[:, 1:] is character occurrence.
            Torch.Tensor : the length of text length [batch_size]

        """

        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.

        batch_max_length += 1
        # +1 means at the first time step add symbol '[GO]', last time step add symbol '[s]'
        batch_text = np.full((len(text), batch_max_length + 1), self.bos, dtype=np.long)
        for i, content in enumerate(text):
            chars = ['[GO]'] + list(content)
            chars.append('[s]')

            if self.with_unknown:
                chars = [self.dict[char] if char in self.dict else self.dict["[UNK]"] for char in chars]
            else:
                try:
                    chars = [self.dict[char] for char in chars]
                except Exception as DictionaryError:
                    raise KeyError from DictionaryError
            batch_text[i, :len(chars)] = chars

        batch_text = torch.tensor(batch_text).long().cuda()
        length = torch.tensor(length).int().cuda()

        return batch_text, length

    def decode(self, text_index, length):
        """
            convert text-index into text-label.
        Args:
            text_index (Torch.tensor): decode text index
            length (Torch.tensor): max text length

        Returns:
            list(str): decode text

        """

        texts = []
        for index, _ in enumerate(length):
            # transfer the model prediction to text
            text = ''.join([self.character[i]
                            for i in text_index[index, :]])
            texts.append(text)
        return texts
