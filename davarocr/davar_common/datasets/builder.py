"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    builder.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
import copy
import platform
from functools import partial
import torch
from torch.utils.data import DataLoader

from mmcv.utils import Registry
from mmcv.utils import build_from_cfg
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.parallel import DataContainer as DC

from mmdet.datasets import DATASETS
from mmdet.models.builder import build
from mmdet.datasets.builder import worker_init_fn
from mmdet.datasets.samplers import DistributedGroupSampler, GroupSampler, DistributedSampler
from mmdet.datasets.pipelines.formating import to_tensor

from .davar_dataset_wrappers import DavarConcatDataset
from .davar_multi_dataset import DavarMultiDataset

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


SAMPLER = Registry('sampler')


def build_sampler(cfg):
    """Build sampler

    Args:
        cfg(mmcv.Config): Sample cfg

    Returns:
        obj: sampler
    """
    return build(cfg, SAMPLER)


def davar_build_dataloader(dataset,
                           samples_per_gpu=1,
                           workers_per_gpu=1,
                           sampler_type=None,
                           num_gpus=1,
                           dist=True,
                           shuffle=True,
                           seed=None,
                           **kwargs):

    """

    Args:
        dataset (Dataset): dataset
        samples_per_gpu (int): image numbers on each gpu
        workers_per_gpu (int): workers each gpu
        sampler_type (optional | dict): sampler parameter
        num_gpus (int): numbers of gpu
        dist (boolean): whether to use distributed mode
        shuffle (boolean): whether to shuffle the dataset
        seed (int): seed number
        **kwargs (None): back parameter

    Returns:
        the training data loader
    """

    rank, world_size = get_dist_info()

    if sampler_type is not None:
        sampler = sampler_type
    else:
        sampler = kwargs.pop('sampler', None)

    cfg_collate = kwargs.pop('cfg_collate', None)

    # if choose distributed sampler
    if dist:
        # whether to shuffle data
        if shuffle:
            if sampler is None:
                # Distributed Group Sampler
                sampler = DistributedGroupSampler(dataset, samples_per_gpu, world_size, rank,)
            else:
                sampler['dataset'] = dataset
                sampler['samples_per_gpu'] = samples_per_gpu

                # build distributed sampler
                sampler = build_sampler(sampler)
        else:
            # distributed sampler
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        if shuffle:
            if sampler is None:
                # Group Sampler
                sampler = GroupSampler(dataset, samples_per_gpu)
            else:
                sampler['dataset'] = dataset
                sampler['samples_per_gpu'] = samples_per_gpu

                # build non-distributed sampler
                sampler = build_sampler(sampler)
        else:
            sampler = None

        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    # combine the training image to mini-batch tensor
    init_fn = partial(worker_init_fn,
                      num_workers=num_workers,
                      rank=rank,
                      seed=seed) if seed is not None else None

    # build data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=multi_frame_collate if cfg_collate == 'multi_frame_collate' else partial(collate, samples_per_gpu=
                                                                                            samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def _concat_dataset(cfg, default_args=None):
    """

    Args:
        cfg (cfg): model config file
        default_args (args): back parameter

    Returns:
        concat all the dataset in config file

    """

    # dataset information, pipeline information, batch setting information
    ann_files = cfg['ann_file']
    img_prefixes = cfg.get('img_prefix', None)
    seg_prefixes = cfg.get('seg_prefix', None)
    proposal_files = cfg.get('proposal_file', None)
    data_types = cfg.get('data_type', None)
    pipeline = cfg.get('pipeline', None)
    batch_ratios = cfg.get('batch_ratios', None)

    # update the parameter of the config
    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['proposal_file'] = proposal_files[i]
        if isinstance(data_types, (list, tuple)):
            data_cfg['data_type'] = data_types[i]
        if isinstance(pipeline, (list, tuple)):
            if isinstance(pipeline[0], (list, tuple)):
                data_cfg['pipeline'] = pipeline[i]
        if isinstance(batch_ratios, (list, tuple)):
            data_cfg['batch_ratios'] = batch_ratios[i]

        # build the dataset
        datasets.append(davar_build_dataset(data_cfg, default_args))

    return DavarConcatDataset(datasets)


def davar_build_dataset(cfg, default_args=None):
    """

    Args:
        cfg (cfg): model config file
        default_args (args): back parameter

    Returns:
        build the dataset for training

    """
    from mmdet.datasets.dataset_wrappers import (ConcatDataset, RepeatDataset,
                                                 ClassBalancedDataset)
    from mmdet.datasets import build_dataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'DavarMultiDataset':
        align_parameters = parameter_align(cfg)
        dataset = DavarMultiDataset(cfg["batch_ratios"],
                                    [davar_build_dataset(c, default_args) for c in align_parameters])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def parameter_align(cfg):
    """ pipeline parameter alignment
    Args:
        cfg (config): model pipeline config

    Returns:

    """
    align_para = list()

    if isinstance(cfg["batch_ratios"], (float, int)):
        batch_ratios = [cfg["batch_ratios"]]
    elif isinstance(cfg["batch_ratios"], (tuple, list)):
        batch_ratios = cfg["batch_ratios"]
    else:
        batch_ratios = list(map(float, cfg["batch_ratios"].split('|')))

    if isinstance(cfg["dataset"]["ann_file"], str):
        cfg["dataset"]["ann_file"] = cfg["dataset"]["ann_file"].split('|')

    if isinstance(cfg["dataset"]["img_prefix"], str):
        cfg["dataset"]["img_prefix"] = cfg["dataset"]["img_prefix"].split('|')

    dataset_num = len(batch_ratios)

    for key, item in cfg["dataset"].items():
        if isinstance(item, list) and isinstance(item[0], list) and len(item) < dataset_num:
            for _ in range(dataset_num - len(item)):
                cfg["dataset"][key].append(item)
        elif isinstance(item, list) and isinstance(item[0], dict):
            temp = []
            for _ in range(dataset_num):
                temp.append(item)
            cfg["dataset"][key] = temp
        elif isinstance(item, list) and len(item) == dataset_num:
            continue
        elif isinstance(item, (int, float)):
            temp = []
            for _ in range(dataset_num):
                temp.append(item)
            cfg["dataset"][key] = temp
        elif isinstance(item, str):
            temp_ = []
            for _ in range(dataset_num):
                temp_.append(item)
            cfg["dataset"][key] = temp_
        else:
            raise TypeError("parameter type error")

    for i in range(dataset_num):
        temp_dict = dict()
        for key, item in cfg["dataset"].items():
            temp_dict[key] = item[i]
        align_para.append(temp_dict)

    return align_para


def multi_frame_collate(batch):
    """
    Args:
        batch (list): one batch data
    Returns:
        dict: collate batch data
    """
    data = dict()
    # this collate func only support batch[0] contains multi instances
    if isinstance(batch[0], list):
        img_meta = []
        img = []
        gt_mask = []
        max_w, max_h = 0, 0
        max_mask_w, max_mask_h = 0, 0

        # calculate the max width and max height to pad
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                size = batch[i][j]['img'].data.size()
                size_mask = batch[i][j]['gt_masks'].data.shape
                if max_w < size[1]:
                    max_w = size[1]
                if max_h < size[2]:
                    max_h = size[2]
                if max_mask_w < size_mask[1]:
                    max_mask_w = size_mask[1]
                if max_mask_h < size_mask[2]:
                    max_mask_h = size_mask[2]

        # pad each img and gt into max width and height
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                img_meta.append(batch[i][j]['img_metas'].data)
                c, w, h = batch[i][j]['img'].data.size()
                tmp_img = torch.zeros((c, max_w, max_h), dtype=torch.float)
                tmp_img[:, 0:w, 0:h] = batch[i][j]['img'].data
                img.append(tmp_img)
                c_mask, w_mask, h_mask = batch[i][j]['gt_masks'].data.shape
                tmp_mask = torch.zeros((c_mask, max_mask_w, max_mask_h), dtype=torch.float)
                mask = to_tensor(batch[i][j]['gt_masks'].data)
                tmp_mask[:, :w_mask, :h_mask] = mask
                gt_mask.append(tmp_mask)

        img = DC([torch.stack(img, dim=0)])
        gt_mask = DC([torch.stack(gt_mask, dim=0)])
        data['img_metas'] = DC([img_meta], cpu_only=True)
        data['img'] = img
        data['gt_masks'] = gt_mask

    else:
        raise "not support type {} of batch".format(type(batch[0]))
    return data