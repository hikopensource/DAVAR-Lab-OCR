"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_checkpoint.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
from mmcv.runner.hooks import HOOKS
from mmcv.runner import LrUpdaterHook
from torch._six import inf


@HOOKS.register_module()
class ReduceonplateauLrUpdaterHook(LrUpdaterHook):
    """ Custom learning rate strategy. Reduce Lr on Plateau."""
    def __init__(self,
                 metric='mAP',
                 mode='max',
                 factor=0.1,
                 patience=10,
                 threshold=1e-5,
                 min_lr=1e-5,
                 **kwargs):
        """
        Args:
            metric (str): evaluation metric used to determine whether it's time to reduce lr.
            mode (str): max/ min of metric.
            factor (float): scale factor to reduce lr.
            patience (int): epoch period of no improvement over metric.
            threshold (float): threshold used to determine best metric.
            min_lr (float): minimum lr.
            kwargs (None): backup parameters

        """
        super().__init__(**kwargs)
        self.metric = metric
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best = None
        self.best_indice = None

        self._reset()

    def _reset(self):
        """
        Reset parameters.
        """
        self.best = inf if self.mode == 'min' else -inf
        self.reduce_num = 0
        self.best_indice = 0

    def is_better(self, value, best):
        """
        Args:
            value (float): current metric value
            best (float): best history metric value
        Returns:
            bool: whether to update best metric
        """
        if self.mode == 'min':
            return value < best - self.threshold

        return value > best + self.threshold

    def get_lr(self, runner, base_lr):
        """
        Args:
            runner (Runner): training runner.
            base_lr (float): base learning rate.
        Returns:
            float: current learning rate.
        """
        if len(getattr(runner, 'eval_hists', [])) > 0:
            current = runner.eval_hists[-1][self.metric]
        else:
            return max(base_lr * self.factor ** self.reduce_num, self.min_lr)

        cur_indice = len(runner.eval_hists) - 1

        if self.is_better(current, self.best):
            self.best = current
            self.best_indice = cur_indice

        if cur_indice - self.best_indice > self.patience:
            self.best_indice = cur_indice
            self.reduce_num += 1
            runner.logger.info('Reduce Lr on Plateau to {}'.format(
                max(base_lr * self.factor ** self.reduce_num, self.min_lr)
            ))

        return max(base_lr * self.factor ** self.reduce_num, self.min_lr)
