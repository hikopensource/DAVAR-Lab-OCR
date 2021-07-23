"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    beams.py
# Abstract       :    Implements of Beams related class utils

# Current Version:    1.0.0
# Date           :    2021-04-30
# Thanks to      :    We borrow the released code from
                      https://github.com/igormq/ctcdecode-pytorch/blob/master/ctcdecode/beams.py
##################################################################################################
"""
import copy

from collections.abc import MutableMapping
from operator import itemgetter

import numpy as np

LOG_0 = -float('inf')
LOG_1 = 0.0


class _Beam:
    def __init__(self):

        self.p = LOG_0
        self.p_b = LOG_0
        self.p_nb = LOG_0

        self.n_p_b = LOG_0
        self.n_p_nb = LOG_0

        self.score = LOG_0
        self.score_lm = LOG_1
        self.score_ctc = LOG_0

        self.state = {}
        self.timesteps = ()

    def step(self):
        self.p_b, self.p_nb = self.n_p_b, self.n_p_nb
        self.n_p_b = self.n_p_nb = LOG_0
        self.score_ctc = np.logaddexp(self.p_b,
                                      self.p_nb)
        self.score = self.score_ctc + self.score_lm

    def __repr__(self):
        return (f'Beam(p_b={self.p_b:.4f}, '
                f'p_nb={self.p_nb:.4f}, '
                f'' f'score={self.score:.4f})')


class Beams(MutableMapping):
    def __init__(self,
                 is_valid=None):
        self.is_valid = is_valid
        self.timestep = 0

        self.beams = {(): _Beam()}
        self.beams[()].p_b = 0
        self.beams[()].score_ctc = 0.0

    def __getitem__(self, key):
        return self.getitem(key)

    def getitem(self,
                key,
                p=None,
                previous_beam=None):
        if key in self.beams:
            beam = self.beams[key]
            if p and p > beam.p:
                beam.p = p
                beam.timesteps = beam.timesteps[:-1] + (self.timestep,)
            return beam

        new_beam = _Beam()

        if previous_beam:
            new_beam.timesteps = previous_beam.timesteps + (self.timestep, )
            new_beam.p = p
            new_beam.state = copy.deepcopy(previous_beam.state)

            if self.is_valid and not self.is_valid(key[-1], new_beam.state):
                return None

        self.beams[key] = new_beam

        return new_beam

    def __setitem__(self,
                    key,
                    value):

        self.beams[key] = value

    def __delitem__(self,
                    key):

        del self.beams[key]

    def __len__(self):
        return len(self.beams)

    def __iter__(self):
        return iter(self.beams)

    def step(self):
        for beam in self.beams.values():
            beam.step()

        self.timestep += 1

    def topk_(self,
              k):
        if len(self.beams) <= k:
            return self

        beams = list(self.beams.items())
        indexes = np.argpartition(
            [-v.score for k, v in beams], k)[:k].tolist()

        self.beams = {k: v
                      for k, v in itemgetter(*indexes)(beams)}

        return self

    def sort(self):
        return sorted(self.beams.items(),
                      key=lambda x: x[1].score,
                      reverse=True)
