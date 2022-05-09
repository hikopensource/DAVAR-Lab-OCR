"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ctc_converter.py
# Abstract       :    Implementations of text-label and text-index of CTC Loss

# Current Version:    1.0.0
# Date           :    2021-04-30
# Thanks to      :    We borrow the released code from http://gitbug.com/clovaai/deep-text-recognition-benchmark
                      for the CTCLabelConverter.
##################################################################################################
"""
import json
import os.path as osp
import math
import torch
import numpy as np

from mmcv.utils import print_log

from davarocr.davar_common.core.builder import CONVERTERS
from .utils.beams import Beams


@CONVERTERS.register_module()
class CTCLabelConverter:
    """Convert between text-label and text-index """
    def __init__(self, character,
                 use_cha_eos=False,
                 cates=9999,
                 with_unknown=False):
        """
            Convert between text-label and text-index
        Args:
            character (str): set of the possible recognition characters dictionary.
            use_cha_eos (bool): use the character define
            cates (int): category number of the character
            with_unknown (bool): whether to encode the characters which are out of the dictionary to ['[UNK]']
        """

        self.eos = 0
        self.with_unknown = with_unknown

        # character dictionary is file format
        if osp.isfile(character):
            with open(character, 'r', encoding='utf-8') as character_file:
                if character.endswith('.json'):
                    # character dictionary is json file
                    print_log("loading user predefined recognition dictionary from json file: "
                              "%s to build the CTC converter !!!" % character)

                    character = json.load(character_file)
                    assert 'char2index' in character
                    self.character = ['' for _ in range(cates)]
                    if use_cha_eos:
                        self.dict = character['char2index']
                        self.eos = self.dict['EOS']
                        for key, value in self.dict.items():
                            self.character[value] = key
                        self.dict['§'] = self.eos
                    else:
                        self.dict = dict()
                        for key, value in character['char2index'].items():
                            self.dict[key] = value + 1
                            self.character[value + 1] = key
                    self.character[self.eos] = '[blank]'

                # character dictionary is txt file
                elif character.endswith('.txt'):
                    print_log("loading user predefined recognition dictionary from txt file: "
                              "%s to build the CTC converter !!!" % character)
                    character = character_file.readline().strip()
                    if self.with_unknown:
                        unk_token = ['[UNK]']
                    else:
                        unk_token = list()
                    dict_character = list(character) + unk_token

                    self.dict = {}
                    for i, char in enumerate(dict_character):
                        # index 0 for the the ['blank'] in CTCLoss
                        self.dict[char] = i + 1

                    self.character = ['[blank]'] + dict_character
                    # supplement for the '[blank]' (index 0)
                else:
                    raise Exception("dictionary file type is not support !!!")
        elif ".json" in character or ".txt" in character:
            # character file does not exist, raise the error
            raise FileNotFoundError("The recognition character file is not existing")
        else:
            raise Exception("dictionary file only support the txt and json file !!!")

        self.max_index = len(self.character) - 1

        print("recognition dictionary %s \t" % str(self.dict).encode(encoding="utf-8").decode(encoding="utf-8"))

    def encode(self, text):
        """
            convert text-label into text-index.
        Args:
            text (list): text labels of each image. [batch_size]

        Returns:
            Torch.Tensor: the training target of the ctc loss. [batch_size x (character_num)].
                text[:, 0] is text length and text[:, 1:] is character occurrence.
            Torch.Tensor: the length of text length [batch_size]

        """

        if isinstance(text[0], list):
            text = ['§' * x[1][0] + x[0] + '§' * x[1][1] for x in text]

        length = [len(s) for s in text]
        text = ''.join(text)

        if self.with_unknown:
            text = [self.dict[char] if char in self.dict else self.dict["[UNK]"] for char in text]
        else:
            try:
                text = [self.dict[char] for char in text]
            except KeyError:
                raise Exception("Dictionary Error, some character not in the predefined recognition dictionary !!!")

        return torch.cuda.IntTensor(text), torch.cuda.IntTensor(length)

    def decode(self, text_index, length, get_before_decode=False):
        """
            convert text-index into text-label.
        Args:
            text_index (Torch.tensor): decode text index
            length (Torch.tensor): max text length
            get_before_decode(bool): whether to deal with the ['eos'] before decode

        Returns:
            list(str): decode text

        """

        if get_before_decode:
            texts = []
            texts2 = []
            index = 0
            for len_ in length:
                t_temp = text_index[index:index + len_]
                char_list = []
                char_list2 = []
                for i in range(len_):
                    if t_temp[i] == self.eos:
                        char_list2.append('_')
                    else:
                        t_temp[i] = min(self.max_index, t_temp[i])
                        char_list2.append(self.character[t_temp[i]])
                    if t_temp[i] != self.eos and (not (i > 0 and t_temp[i - 1] == t_temp[i])):
                        # removing repeated characters and blank.
                        t_temp[i] = min(self.max_index, t_temp[i])
                        char_list.append(self.character[t_temp[i]])
                text = ''.join(char_list)
                text2 = ''.join(char_list2)

                texts.append(text)
                texts2.append(text2)
                index += len_
            return texts, texts2

        texts = []
        index = 0
        for len_ in length:
            t_temp = text_index[index:index + len_]
            char_list = []
            for i in range(len_):
                if t_temp[i] != self.eos and (not (i > 0 and t_temp[i - 1] == t_temp[i])):
                    # remove the redundant text and blank
                    t_temp[i] = min(self.max_index, t_temp[i])
                    char_list.append(self.character[t_temp[i]])
            text = ''.join(char_list)
            texts.append(text)
            index += len_
        return texts

    def decode_perh(self, pred, score_map):
        """

        Args:
            pred (Torch.tensor): model prediction text index
            score_map (Torch.tensor): score map

        Returns:
            list(str): decoded text of the model prediction

        """
        batch_size = pred.size(0)
        length = pred.size(3)
        score_map = score_map.squeeze(1)  # n h w
        score_map = score_map.permute(0, 2, 1)  # n w h
        _, h_indexes = score_map.max(2)  # h dimension max value

        texts = []
        for b in range(batch_size):
            char_list = []
            for i in range(length):
                h_index = h_indexes[b, i]
                _, t = pred[b, :, h_index, :].max(0)  # channel dimension max value
                if t[i] != self.eos and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            texts.append(text)
        return texts

    def ctc_beam_search_decoder(self,
                                log_probs_seq,
                                lm_scorer=None,
                                beam_size=66,
                                blank=96,
                                cutoff_prob=1.0,
                                cutoff_top_n=2):
        """
        Performs prefix beam search on the output of a CTC network.
        Args:
            log_probs_seq (Torch.tensor): The log probabilities. Should be a 2D array (time_steps x alphabet_size)
            lm_scorer (func): Language model function. Should take as input a string and output a probability.
            beam_size (int): The beam width. Will keep the `beam_size` most likely candidates at each time_step.
            blank (int): Blank label index
            cutoff_prob: Cutoff probability for pruning. Defaults to `1.0`, meaning no pruning >>>ratio should be
            more comprehensible
            cutoff_top_n: Cutoff number for pruning.
        Returns:
            list(str): The decoded CTC output.
        """
        T, V = log_probs_seq.shape
        log_cutoff_prob = math.log(cutoff_prob)
        cutoff_top_n = min(cutoff_top_n, V) if cutoff_top_n else V

        beams = Beams(is_valid=lm_scorer.is_valid if lm_scorer else None)

        for t in range(T):

            log_probs = log_probs_seq[t]

            curr_beams = list(beams.sort())

            # A default dictionary to store the next step candidates.
            num_prefixes = len(curr_beams)

            min_cutoff = -float('inf')
            full_beam = False

            if lm_scorer:
                # score - beta # it will not insert a new word or character
                min_cutoff = curr_beams[-1][-1].score_ctc + log_probs[blank]
                full_beam = num_prefixes == beam_size

            # Prunning step
            pruned_indexes = torch.arange(len(log_probs)).tolist()
            if log_cutoff_prob < 0.0 or cutoff_top_n < V:
                idxs = torch.argsort(log_probs, descending=True)
                n_idxs = min(cutoff_top_n, V)
                pruned_indexes = idxs[:n_idxs].tolist()

            for token_index in pruned_indexes:

                p = log_probs[token_index].item()

                # The variables p_b and p_nb are respectively the probabilities for the prefix given that it ends in a
                # blank and does not end in a blank at this time step.
                for prefix, beam in curr_beams:
                    p_b, p_nb = beam.p_b, beam.p_nb

                    if full_beam and p + beam.score_ctc < min_cutoff:
                        break

                    # If we propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
                    if token_index == blank:
                        beam.n_p_b = np.logaddexp(beam.n_p_b,
                                                  beam.score_ctc + p)
                        continue

                    # Extend the prefix by the new character s and add it to the beam[' Only'] the
                    # probability of not ending in blank gets updated.
                    last_token_index = prefix[-1] if prefix else None

                    if token_index == last_token_index:
                        # If s is repeated at the end we also update the unchanged prefix. This is the merging case.
                        beam.n_p_nb = np.logaddexp(beam.n_p_nb, p_nb + p)

                    n_prefix = prefix + (token_index, )

                    # Must update state for prefix search
                    n_beam = beams.getitem(n_prefix,
                                           p=p,
                                           previous_beam=beam)
                    if not n_beam:
                        continue

                    n_p_nb = n_beam.n_p_nb

                    if token_index == last_token_index and p_b > -float('inf'):
                        # We don't include the previous probability of not ending in blank (p_nb)
                        # if s is repeated at the end.
                        # The CTC algorithm merges characters not separated by a blank.
                        n_p_nb = np.logaddexp(n_p_nb, p_b + p)
                    elif token_index != last_token_index:
                        n_p_nb = np.logaddexp(n_p_nb, beam.score_ctc + p)

                    if lm_scorer:
                        # LM scorer has access and updates the state variable
                        p_lm = lm_scorer(n_prefix, n_beam.state)
                        n_beam.score_lm = beam.score_lm + p_lm

                    n_beam.n_p_nb = n_p_nb

            # Update the probabilities
            beams.step()
            # Trim the beam before moving on to the next time-step.
            beams.topk_(beam_size)

        # score the eos
        if lm_scorer:
            for prefix, beam in beams.items():
                if prefix:
                    p_lm = lm_scorer(prefix,
                                     beam.state,
                                     eos=True)
                    beam.score_lm += p_lm
                    beam.score = beam.score_ctc + beam.score_lm

        # Return the top beam_size -log probabilities without the lm scoring
        result = [(-beam.score, p, beam.timesteps)
                  for p, beam in beams.sort()]

        text_res = ''.join([self.character[s]
                            for s in result[0][1]])

        return text_res
