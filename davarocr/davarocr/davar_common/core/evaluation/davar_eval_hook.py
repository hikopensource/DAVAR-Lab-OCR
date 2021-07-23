"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_eval_hook.py
# Abstract       :    Implementation of the evaluation hook of davar group.

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
import os.path as osp
from math import inf

import torch
import torch.distributed as dist
from mmdet.core import EvalHook, DistEvalHook
from mmcv.utils import print_log

from ...apis import single_gpu_test, multi_gpu_test


class DavarEvalHook(EvalHook):
    """ Customized evaluation hook, support for evaluate by iterations."""
    greater_keys = ['mAP', 'AR', 'accuracy', 'hmean']
    init_value_map = {'greater': -inf, 'less': inf}
    less_keys = ['loss', 'NED']

    def __init__(self,
                 dataloader,
                 model_type='DETECTOR',
                 start=None,
                 start_iter=None,
                 interval=1,
                 iter_interval=-1,
                 eval_mode="general",
                 by_epoch=True,
                 by_iter=False,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        """
        Args:
            dataloader (nn.dataloader): validation dataloader
            model_type (str): model type, including ['DETECTOR', 'RECOGNIZOR', 'SPOTTER']
            start (int): Evaluation starting epoch
            start_iter (float): percentage of the total iteration to start evaluation
            interval (int): The epoch saving period.
            iter_interval (int): The iteration saving period.
            by_epoch (bool): Saving checkpoints by epoch
            by_iter (bool): Saving checkpoints by iteration
            save_best (str): If a metric is specified, it would measure the best checkpoint during evaluation.
            rule (str): Comparison rule for best score. 'greater' or 'less'
            **eval_kwargs (None): backup parameter
        """

        super().__init__(dataloader=dataloader, start=start, interval=interval,
                         by_epoch=by_epoch, save_best=save_best, rule=rule, **eval_kwargs)

        # evaluation interval including epoch and iteration
        self.interval = interval
        self.iter_interval = iter_interval
        self.model_type = model_type

        self.by_iter = by_iter
        self.start_iter = start_iter

        assert eval_mode in ["general", "lightweight"], \
            'eval_mode mode should be in ["general", "lightweight"], ''but found' + eval_mode
        self.eval_mode = eval_mode

        self.eval_kwargs = eval_kwargs

        if self.save_best is not None:
            self._init_rule(rule, self.save_best)

    def evaluation_iteration_flag(self, runner):
        """
        Args:
            runner (Runner): the controller of the training process

        Returns:

        """
        if self.start_iter is None:
            if not self.every_n_iters(runner, self.iter_interval):
                return False
        elif (runner.iter + 1) < int(self.start_iter * runner.max_iters):
            return False
        else:
            if (runner.iter + 1 - int(self.start_iter * runner.max_iters)) % self.iter_interval:
                return False
        return True

    def after_train_iter(self, runner):
        """
        Args:
            runner (Runner): the controller of the training process

        Returns:

        """
        if not self.by_iter or self.iter_interval == -1 or not self.evaluation_iteration_flag(runner):
            return

        # model inference
        results = single_gpu_test(runner.model, self.dataloader, show=False, model_type=self.model_type)

        # change the training state
        runner.model.train()

        # calculate the evaluation metric
        key_score = self.evaluate(runner, results)

        if self.save_best:
            if self.eval_mode == "general":
                # official MMDetection evaluation
                self.save_best_checkpoint(runner, key_score)
            elif self.eval_mode == "lightweight":
                # davar lightweight evaluation
                self.light_save_best_checkpoint(runner=runner, key_score=key_score, train_type="iter")
            else:
                raise NotImplementedError('Current version only support "general" and "lightweight" mode !!!')

    def after_train_epoch(self, runner):
        """
        Args:
            runner (Runner): the controller of the training process

        Returns:

        """
        if not self.by_epoch or not self.evaluation_flag(runner):
            return

        # model inference
        results = single_gpu_test(runner.model, self.dataloader, show=False, model_type=self.model_type)

        # calculate the evaluation metric
        key_score = self.evaluate(runner, results)

        if self.save_best:
            eval_hists = getattr(runner, 'eval_hists', [])
            eval_hists.append({self.save_best: key_score})
            runner.eval_hists = eval_hists

            if self.eval_mode == "general":
                self.save_best_checkpoint(runner, key_score)
            elif self.eval_mode == "lightweight":
                self.light_save_best_checkpoint(runner=runner, key_score=key_score, train_type="epoch")
            else:
                raise NotImplementedError('Current version only support general and lightweight mode !!!')

    def save_best_checkpoint(self, runner, key_score):
        """
        Args:
            runner (Runner): the controller of the training process
            key_score (float): current evaluation result

        Returns:

        """
        best_score = runner.meta['hook_msgs'].get('best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score

            if 'last_ckpt' in runner.meta['hook_msgs']:
                last_ckpt = runner.meta['hook_msgs']['last_ckpt']
            else:
                if self.by_epoch:
                    # update the last_ckpt
                    runner.meta['hook_msgs']['last_ckpt'] = runner.epoch + 1
                    last_ckpt = runner.epoch + 1
                else:
                    runner.meta['hook_msgs']['last_ckpt'] = runner.iter + 1
                    last_ckpt = runner.iter + 1
            runner.meta['hook_msgs']['best_ckpt'] = last_ckpt

            # update the best information in logger
            time_stamp = runner.epoch + 1 if self.by_epoch else runner.iter + 1
            runner.meta['hook_msgs']['best_timestamp'] = time_stamp

        print_log('Now best checkpoint is epoch_{}.pth. Best {} is {}'.format(
            runner.meta['hook_msgs'].get("best_timestamp", 0), self.key_indicator, best_score), logger=runner.logger)

    def light_save_best_checkpoint(self, runner, key_score, train_type):
        """
        Args:
            runner (Runner): the controller of the training process
            key_score (str): Key indicator to determine the comparison rule.
            train_type (str): training type, including["epoch", "iter"]

        Returns:

        """

        best_score = runner.meta['hook_msgs'].get('best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score
            if 'last_ckpt' not in runner.meta['hook_msgs']:
                # initialize the last_ckpt parameters in logger
                if train_type == "iter":
                    runner.meta['hook_msgs']['last_ckpt'] = runner.iter + 1
                else:
                    runner.meta['hook_msgs']['last_ckpt'] = runner.epoch + 1

            if 'ckpt_name' not in runner.meta['hook_msgs']:
                # update the best pth file name during the training process
                if train_type == "iter":
                    runner.meta['hook_msgs']['ckpt_name'] = "Iter_" + str(runner.iter + 1)
                else:
                    runner.meta['hook_msgs']['ckpt_name'] = "Epoch_" + str(runner.epoch + 1)

            last_ckpt = runner.meta['hook_msgs']['last_ckpt']
            runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
            best_ckpt_name = runner.meta['hook_msgs']['ckpt_name']
            print_log(
                'Now best checkpoint is {}.pth. Best {} is {}'.format(best_ckpt_name, self.key_indicator,
                                                                      best_score), logger=runner.logger)


class DavarDistEvalHook(DavarEvalHook, DistEvalHook):
    """ Customized evaluation hook, support for evaluate by iterations. """
    def __init__(self, dataloader,
                 start=None,
                 start_iter=None,
                 model_type="DETECTOR",
                 eval_mode="general",
                 interval=1,
                 iter_interval=-1,
                 by_epoch=True,
                 by_iter=False,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        """
        Args:
            dataloader (nn.dataloader): validation dataloader
            model_type (str): model type, including ['DETECTOR', 'RECOGNIZOR', 'SPOTTER']
            start (int): Evaluation starting epoch
            start_iter (float): percentage of the total iteration to start evaluation
            eval_mode (str): model evaluation type
            interval (int): The epoch saving period.
            iter_interval (int): The iteration saving period.
            by_epoch (bool): Saving checkpoints by epoch
            by_iter (bool): Saving checkpoints by iteration
            save_best (str): If a metric is specified, it would measure the best checkpoint during evaluation.
            rule (str): Comparison rule for best score.
            **eval_kwargs (None): backup parameter
        """
        super().__init__(dataloader=dataloader, model_type=model_type,
                         start=start, start_iter=start_iter, interval=interval,
                         iter_interval=iter_interval, eval_mode=eval_mode,
                         by_epoch=by_epoch, by_iter=by_iter, save_best=save_best,
                         rule=rule, **eval_kwargs)

        # evaluation interval including epoch and iteration
        self.iter_interval = iter_interval
        self.by_iter = by_iter

        self.start_iter = start_iter
        self.model_type = model_type

        assert eval_mode in ["general", "lightweight"], \
            'eval_mode mode should be in ["general", "lightweight"], ''but found' + eval_mode
        self.eval_mode = eval_mode

        self.eval_kwargs = eval_kwargs

        if self.save_best is not None:
            self._init_rule(rule, self.save_best)

    def after_train_epoch(self, runner):
        """
        Args:
            runner (Runner): the controller of the training process

        Returns:

        """
        if not self.by_epoch or not self.evaluation_flag(runner):
            return

        if self.broadcast_bn_buffer:
            self._broadcast_bn_buffer(runner)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # model inference
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            model_type=self.model_type)

        best_score = torch.full((1,), 0., dtype=torch.float, device='cuda')

        if runner.rank == 0:
            # calculate the evaluation metric
            key_score = self.evaluate(runner, results)
            if self.save_best:
                best_score[0] = key_score

                if self.eval_mode == "general":
                    self.save_best_checkpoint(runner, key_score)
                elif self.eval_mode == "lightweight":
                    self.light_save_best_checkpoint(runner=runner, key_score=key_score, train_type="epoch")
                else:
                    raise NotImplementedError('Current version only support "general" and "lightweight" mode !!!')

        # broadcast from best score to all gpus
        dist.broadcast(best_score, 0)
        dist.barrier()
        eval_hists = getattr(runner, 'eval_hists', [])
        eval_hists.append({self.save_best: best_score[0]})
        runner.eval_hists = eval_hists

    def after_train_iter(self, runner):
        """
        Args:
            runner (Runner): the controller of the training process

        Returns:

        """
        if not self.by_iter or self.iter_interval == -1 or not self.evaluation_iteration_flag(runner):
            return
        if self.broadcast_bn_buffer:
            self._broadcast_bn_buffer(runner)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # model inference
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            model_type=self.model_type
        )

        # change the training state
        runner.model.train()

        if runner.rank == 0:
            # calculate the evaluation metric
            key_score = self.evaluate(runner, results)

            if self.save_best:
                if self.eval_mode == "general":
                    # official MMDetection evaluation
                    self.save_best_checkpoint(runner, key_score)
                elif self.eval_mode == "lightweight":
                    # DavarOCR lightweight evaluation
                    self.light_save_best_checkpoint(runner=runner, key_score=key_score, train_type="iter")
                else:
                    raise NotImplementedError('Current version only support "general" and "lightweight" mode !!!')
