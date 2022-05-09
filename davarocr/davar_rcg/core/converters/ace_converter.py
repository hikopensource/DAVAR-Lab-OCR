"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ace_converter.py
# Abstract       :    Implementations of text-label and text-index of ACE Loss

# Current Version:    1.0.0
# Date           :    2021-04-30
# Thanks to      :    We borrow the released code from https://github.com/summerlvsong/Aggregation-Cross-Entropy
                      for the ACELabelConverter.
##################################################################################################
"""
import os.path as osp

import torch
from mmcv.utils import print_log

from davarocr.davar_common.core.builder import CONVERTERS


@CONVERTERS.register_module()
class ACELabelConverter:
    """Convert between text-label and text-index, ACE Loss Converter in Ref [1]

       Ref: [1] Aggregation Cross-Entropy for Sequence Recognition. CVPR-2019
    """
    def __init__(self, character,
                 with_unknown=False):
        """
            Convert between text-label and text-index
        Args:
            character (str): set of the possible recognition characters dictionary.
            with_unknown (bool): whether to encode the characters which are out of the dictionary to ['[UNK]']
        """

        self.with_unknown = with_unknown
        # character dictionary is file format
        if osp.isfile(character):
            with open(character, 'r', encoding='utf-8') as character_file:

                # character dictionary is txt file
                if character.endswith('.txt'):
                    print_log("loading user predefined recognition dictionary from txt file: "
                              "%s to build the ACE converter !!!" % character)
                    character = character_file.readline().strip()

                    # [GO] for the start token of the attention decoder.
                    # [s] for end-of-sentence token.
                    list_token = ['[PAD]']
                    if self.with_unknown:
                        unk_token = ['[UNK]']
                    else:
                        unk_token = list()
                    # ['[s]','[UNK]','[PAD]','[GO]']
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
            Torch.Tensor : the training target. [batch_size x (character_num)].
                text[:, 0] is text length and text[:, 1:] is character occurrence.
            Torch.Tensor : the length of text length [batch_size]

        """

        length = [len(s) for s in text]  # +1 for [PAD] of the sentence.

        batch_max_length += 1

        # batch_text is padded with [PAD] token.
        batch_text = torch.cuda.LongTensor(len(text), len(self.character)).fill_(0)

        for i, t_ in enumerate(text):
            text = [item for item in list(t_)]
            if self.with_unknown:
                text = [self.dict[char] if char in self.dict.keys() else self.dict["[UNK]"] for char in text]
            else:
                try:
                    text = [self.dict[char] for char in text]
                except Exception as DictionaryError:
                    raise KeyError from DictionaryError
            text_cnt = torch.cuda.LongTensor(len(self.character) - 1).fill_(0)
            for ln_ in text:
                text_cnt[ln_ - 1] += 1  # label construction for ACE
            batch_text[i][1:] = text_cnt
        batch_text[:, 0] = torch.cuda.IntTensor(length)
        return batch_text, torch.cuda.IntTensor(length)

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
            text = ''.join([self.character[i] for i in text_index[index, text_index[index] != 0]])
            texts.append(text)
        return texts
