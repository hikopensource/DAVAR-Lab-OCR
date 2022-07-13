"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    train.py
# Abstract       :    The common training api for davarocr,
                       Support for DETECTOR, RECOGNIZOR, SPOTTER, INFO_EXTRACTOR, etc.

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
import warnings

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv.utils import build_from_cfg
from mmdet.datasets import replace_ImageToTensor

from ..utils import get_root_logger
from ..datasets import davar_build_dataloader, davar_build_dataset
from ..core import DavarDistEvalHook, DavarEvalHook


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """ The common train api for all ocr models

    Args:
        model (nn,Module):  Model to be trained.
        dataset (nn.Dataloader): Pytorch data loader.
        cfg(obj): Model configuration
        distributed (boolean): whether to start distributed training.
        validate(boolean): whether to open online validation
        timestamp(int): runners timestamp indicator
        meta(dict): prepared meta information for runner.
    """
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                'Got "imgs_per_gpu"={} and "samples_per_gpu"={},"imgs_per_gpu"={} '
                'is used in this experiments'.format(
                    cfg.data.imgs_per_gpu, cfg.data.samples_per_gpu, cfg.data.imgs_per_gpu))
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                '{} in this experiments'.format(cfg.data.imgs_per_gpu))
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    cfg_sampler = cfg.data.get("sampler", None)

    data_loaders = [
        davar_build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            cfg_sampler,       # Allow to customize sampler
            len(cfg.gpu_ids),  # cfg.gpus will be ignored if distributed
            dist=distributed,
            seed=cfg.seed) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:

        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # in case the test dataset is concatenated
            val_pipeline = cfg.data.val.get("pipeline", cfg.data.val.dataset.get("pipeline", None))
            # supported multi dataset with different validation pipelines
            if isinstance(val_pipeline[0], dict):
                cfg.data.val.pipeline = replace_ImageToTensor(val_pipeline)
            elif isinstance(val_pipeline[0], list):
                cfg.data.val.pipeline = [
                    replace_ImageToTensor(this_pipeline) for this_pipeline in val_pipeline]

        val_dataset = davar_build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = davar_build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            sampler_type=None,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DavarDistEvalHook if distributed else DavarEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='HIGH')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow)
