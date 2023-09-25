"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tokenizer.py
# Abstract       :    tokenize gt_texts in character level and pad to max length.

# Current Version:    1.0.0
# Date           :    2020-05-31
# Current Version:    1.0.1
# Date           :    2022-12-12
##################################################################################################
"""
import copy
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CharPadTokenize():
    """Tokenize texts in characters and return their indexes ( padded if required)."""

    def __init__(self, vocab, targets, max_length=None, map_target_prefix=None):
        """
        Args:
            vocab (str): vocab.txt
            targets (list): tokenize on which target(s)
            max_length (int | None): if max_length is None, do not pad texts; otherwise, pad to max_length.
            map_target_prefix (None| str): if map_target_prefix is None, replace origin target; otherwise, create new
                target startswith map_target_prefix.
        """
        self.vocab = vocab
        self.targets = targets
        self.max_length = max_length
        self.map_target_prefix = map_target_prefix

        if self.vocab is not None:
            with open(self.vocab, 'r', encoding='utf8') as read_f:
                all_words = read_f.readline().strip()
            all_words = list(all_words)
        else:
            all_words = []

        default_token = ['[PAD]', '[UNK]']
        self.character = default_token + all_words

        # default 0 to pad
        self.word2idx = {char: idx for idx, char in enumerate(self.character)}

    def __call__(self, results):
        """Forward process, including tokenization and (optional) padding.

        Args:
            results (dict): temporary results of data pipelines.

        Returns:
            dict: overwrite or add new k-v pairs in this process.
        """
        for key in self.targets:
            per_target = copy.deepcopy(results[key])
            per_target_token = []
            for per_line in per_target:
                tmp_per_line = [self.word2idx[per_char] if per_char in self.word2idx.keys() else self.word2idx['[UNK]']
                                for per_char in per_line]
                # pad to max length if required
                if self.max_length is not None:
                    if len(tmp_per_line) > self.max_length:
                        tmp_per_line = tmp_per_line[:self.max_length]
                    else:
                        tmp_per_line.extend([self.word2idx['[PAD]']] * (self.max_length - len(tmp_per_line)))
                per_target_token.append(np.array(tmp_per_line))

            # add map_target to results if required.
            if self.map_target_prefix is not None:
                results[self.map_target_prefix + key] = np.array(per_target_token)
            else:
                results[key] = np.array(per_target_token)

        return results

    def __repr__(self):
        """Return descriptions of this class."""
        return self.__class__.__name__ + '(vocab={}, target={})'.format(self.vocab, ''.join(self.target))
