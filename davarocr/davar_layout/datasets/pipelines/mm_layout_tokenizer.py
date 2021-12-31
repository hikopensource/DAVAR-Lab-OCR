"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mm_layout_tokenizer.py
# Abstract       :    Tokenizer used in multimodal layout analysis.

# Current Version:    1.0.0
# Date           :    2020-12-06
##################################################################################################
"""
import os

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CharTokenize():
    """Original Character tokenizer.
    """
    def __init__(self, vocab, targets, add_space=True):
        """Generating char index.

        Args:
            vocab (str): vocab.txt
            targets (list): tokenize on which target(s)

        """
        self.vocab = vocab
        self.targets = targets

        if os.path.exists(self.vocab):
            all_words = open(self.vocab, 'r', encoding='utf8').readlines()

            # default 0 to background
            self.word2idx = {char.strip(): idx + 1 for idx, char in enumerate(all_words)}
            self.word2idx["UNKNOWN"] = len(all_words) + 1

            if add_space:
                self.word2idx["SEP"] = len(all_words) + 2

    def __call__(self, results):
        """ Main process.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: output data flow.
        """
        for key in self.targets:
            per_target = results[key]
            per_target_token = []
            for per_line in per_target:
                per_line = ["SEP" if per_char in [' ', '\n', '\t'] else per_char for per_char in per_line]
                per_target_token.append([self.word2idx[per_char] if per_char in self.word2idx.keys() else \
                                             self.word2idx["UNKNOWN"] for per_char in per_line])

            results[key] = per_target_token

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(vocab={}, target={})'.format(self.vocab, ''.join(self.target))
