"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    distill_runner.py
# Abstract       :    mmcv runner for distillation

# Current Version:    1.0.0
# Date           :    2021-07-07
##################################################################################################
"""
from mmcv.runner.builder import RUNNERS
from mmcv.runner.epoch_based_runner import EpochBasedRunner

@RUNNERS.register_module()
class DistillRunner(EpochBasedRunner):
    def run_iter(self, data_batch, train_mode, **kwargs):
        # Adding runner to data_batch
        data_batch.update({"runner" : self})
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs