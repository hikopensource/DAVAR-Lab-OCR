"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_checkpoint.py
# Abstract       :    Implementation of the checkpoint hook of davar group.

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
import os
import torch

from mmcv.runner.dist_utils import allreduce_params
from mmcv.runner.hooks import HOOKS
from mmcv.runner import CheckpointHook


@HOOKS.register_module()
class DavarCheckpointHook(CheckpointHook):
    """ Customized Checkpoint Hook, support to only save nearest and best checkpoints"""
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    greater_keys = ['accuracy', 'hmean', 'mAP', 'macro_f1', 'bbox_mAP', 'avg_f1', 'img_level_edge_acc', 'text_acc',
                    'line_acc', 'total_order_acc', 'hard_f1', 'tree_f1']
    less_keys = ['NED']

    def __init__(self,
                 interval=1,
                 iter_interval=-1,
                 by_epoch=True,
                 by_iter=False,
                 metric="accuracy",
                 rule="greater",
                 init_metric=0,
                 save_optimizer=True,
                 out_dir=None,
                 save_mode="general",
                 model_milestone=0,
                 max_keep_ckpts=-1,
                 save_last=True,
                 sync_buffer=False,
                 **kwargs):
        """
        Args:
            interval (int): The epoch saving period.
            iter_interval (int): The iteration saving period.
            by_epoch (bool): Saving checkpoints by epoch
            by_iter (bool): Saving checkpoints by iteration
            epoch_metric (str): the epoch metric compare during save best checkpoint
            iter_metric (str): the iteration metric compare during save best checkpoint
            rule (str): Comparison rule for best score.
            init_metric (float): initialize the metric
            save_optimizer (bool): Whether to save optimizer state_dict in the checkpoint.
            out_dir (str): The directory to save checkpoints.
            save_mode (str): save mode, including ["general", 'lightweight']
            model_milestone (float): the percentage of the total training process to start save checkpoint
            max_keep_ckpts (int): The maximum checkpoints to keep.
            save_last (bool): Whether to force the last checkpoint to be saved regardless of interval.
            sync_buffer (bool): Whether to synchronize buffers in different gpus
            **kwargs (None): backup parameter
        """

        super().__init__(interval, by_epoch, save_optimizer, out_dir, max_keep_ckpts, save_last, sync_buffer, **kwargs)

        self.iter_interval = iter_interval

        self.by_iter = by_iter

        assert save_mode in ["general", 'lightweight'], 'save mode should be general and lightweight, but found' + \
                                                        save_mode
        self.save_mode = save_mode

        self.model_milestone = model_milestone

        assert metric in self.greater_keys or metric in self.less_keys, \
            'epoch_metric mode should be in ["accuracy", "NED"], ''but found' + metric

        self.metric = metric
        self.init_metric = init_metric
        self.save_type = None
        self.compare_func = None
        self.rule = rule
        if self.save_mode == "lightweight":
            self.davar_rule(self.rule)
        self.sync_buffer = sync_buffer

    def davar_rule(self, rule):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Args:
            rule (str | None): Comparison rule for best score.

        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def after_train_epoch(self, runner):
        """
        Args:
            runner (Runner): the controller of the training process

        Returns:

        """

        self.save_type = 'epoch'
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval) or (self.save_last and self.is_last_epoch(runner)):
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            if self.save_mode == "general":
                if (runner.epoch + 1) > int(runner.max_epochs * self.model_milestone):
                    runner.logger.info(
                        f'Saving checkpoint at {runner.epoch + 1} epochs')
                    self._save_checkpoint(runner)
            elif self.save_mode == "lightweight":
                self._davar_save_checkpoint(runner, self.save_type)
            else:
                raise NotImplementedError("Only support the ['general', 'lightweight'] save mode!!")

    def after_train_iter(self, runner):
        """
        Args:
            runner (Runner): the controller of the training process

        Returns:

        """

        self.save_type = 'iter'

        if not self.by_iter or self.iter_interval == -1:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(runner, self.iter_interval) or (self.save_last and self.is_last_iter(runner)):
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            if self.save_mode == "general":
                if (runner.iter + 1) > int(runner.max_iters * self.model_milestone):
                    runner.logger.info(
                        f'Saving checkpoint at {runner.iter + 1} iterations')
                    self._save_checkpoint(runner)
            elif self.save_mode == "lightweight":
                self._davar_save_checkpoint(runner, self.save_type)
            else:
                raise NotImplementedError("Only support the ['general', 'lightweight'] save mode!!")

    def _davar_save_checkpoint(self, runner, save_type):
        """
        Args:
            runner (Runner): the controller of the training process
            save_type (str): save type, including["epoch", "iter"]

        Returns:

        """

        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = os.path.join(
                self.out_dir, cur_ckpt_filename)

        if not self.out_dir:
            self.out_dir = runner.work_dir

        if save_type == "epoch":
            self.init_metric = save_best_checkpoint(runner,
                                                    self.metric,
                                                    self.init_metric,
                                                    save_type,
                                                    self.model_milestone,
                                                    self.out_dir,
                                                    self.save_optimizer,
                                                    self.save_last,
                                                    self.is_last_epoch,
                                                    self.compare_func)
        if save_type == "iter":
            self.init_metric = save_best_checkpoint(runner,
                                                    self.metric,
                                                    self.init_metric,
                                                    save_type,
                                                    self.model_milestone,
                                                    self.out_dir,
                                                    self.save_optimizer,
                                                    self.save_last,
                                                    self.is_last_iter,
                                                    self.compare_func)


def save_best_checkpoint(runner,
                         metric_name,
                         init_metric,
                         save_type,
                         model_milestone,
                         out_dir,
                         save_optimizer,
                         save_last,
                         last_flag,
                         compare_func):
    """
    Args:
        runner (Runner): the controller of the training process
        metric_name (str): compared metric name
        init_metric (float): the iteration metric compare during save best checkpoint
        save_type (str): save type, including["epoch", "iter"]
        model_milestone (float): the percentage of the total training process to start save checkpoint
        out_dir (str): The directory to save checkpoints.
        save_optimizer (bool): whether to save the optimizer
        save_last (bool): whether to save the last epoch or iteration
        last_flag (function): judge the last epoch or last iteration function
        compare_func (function): compared function

    Returns:
        float: update the best result compared to the current evaluation result

    """

    if save_type == "epoch":
        time_flag = runner.epoch + 1
        total_flag = runner.max_epochs
    elif save_type == "iter":
        time_flag = runner.iter + 1
        total_flag = runner.max_iters
    else:
        raise NotImplementedError("Only support the save checkpoint by iteration or epoch !!")

    if time_flag > int(total_flag * model_milestone):
        if metric_name in runner.log_buffer.output:
            cur_metric = runner.log_buffer.output[metric_name]

            # Current performance is greater than historical performance
            if torch.cuda.current_device() == 0:
                if compare_func(cur_metric, init_metric):
                    runner.logger.info("current_metric {} is much better than previous best_metric"
                                       " {}, saving best model !".format(cur_metric, init_metric))
                    runner.logger.info(
                        f'Saving checkpoint at {save_type}_{time_flag}.pth')
                    init_metric = cur_metric
                    best_info = {
                        save_type: time_flag,
                        'best_score': cur_metric,
                                 }
                    runner.meta.update(best_info)
                    save_name = 'Best_checkpoint.pth'
                    runner.save_checkpoint(out_dir, filename_tmpl=save_name,
                                           save_optimizer=save_optimizer,
                                           meta=runner.meta,
                                           create_symlink=False)

            # save the latest model
            if torch.cuda.current_device() == 0 and save_last and last_flag(runner):
                runner.save_checkpoint(out_dir, filename_tmpl='latest_model.pth',
                                       save_optimizer=save_optimizer,
                                       meta=None,
                                       create_symlink=False)

    return init_metric
