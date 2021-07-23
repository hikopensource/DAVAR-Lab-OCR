"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_sampler.py
# Abstract       :    Implementation of the training sampler of davar group.

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
from __future__ import division
import math
import numpy as np
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler

from mmcv.runner import get_dist_info

from ...datasets import SAMPLER


def batch_sampler(sample_type, epoch,
                  mode, num_replicas,
                  num_samples, batch_ratios,
                  group_samples, samples_per_gpu,
                  num_batches, index_unused,
                  shuffle, total_size):
    """
    Args:
        sample_type (str): type of the sampler. including dist(distributed) and non-dist(default)
        epoch (int|None): distributed mode random seed number
        mode (int): sampler mode, including [-1, 0, 1, 2]
                    # model 0：Balance in batch, calculate the epoch according to the first iterative data set
                    # model 1：Balance in batch, calculate the epoch according to the last iterative data set
                    # model 2：Balance in batch, record unused data
                    # model -1：Each dataset is directly connected and shuffled
        num_replicas(int): gpu numbers
        num_samples (int): total images in each epoch
        batch_ratios (list|np.array): use ratio on each dataset in each batch
        group_samples (list|np.array): image numbers of each dataset in each batch
        samples_per_gpu (int): image numbers in each gpu
        num_batches (int): batch numbers of each epoch
        index_unused (list): record the used sample during training process
        shuffle (bool): whether to shuffle data
        total_size (int|None): total sample numbers

    Returns:
        list(int): sampled images index list
    """

    if sample_type == "dist":
        np.random.seed(epoch)
    else:
        num_replicas = 1

    indices = list()
    indices_start = 0
    dataset_count = 0

    if mode == 1 or mode == 2 or not mode:
        # batch ratio of each dataset and image numbers of each dataset in one batch
        # example. (0.1,100),(0.5,200),(0.4,40) batch_size:10(0.1:0.4:0.5)
        for batch_ratio, num_group_samples in zip(batch_ratios, group_samples):
            # calculate the used data numbers of each dataset, 10,50,40
            # num_batches means batch numbers of each epoch, num_batches=10
            num_used = int(round(batch_ratio * samples_per_gpu) * num_batches) * num_replicas
            if mode < 2:
                # calculate the used sample index
                group_indices = np.arange(indices_start, indices_start + num_group_samples)

                # shuffle the used sample index
                if shuffle:
                    np.random.shuffle(group_indices)
                if not mode:
                    # (0,10) (100,150) (300,340), rearrange 1*10 5*10 4*10 format
                    # select the index and rearrange
                    indices_used = group_indices[:num_used].reshape(-1, num_batches)
                    indices.append(indices_used)        # append to the total list
                    indices_start += num_group_samples  # jump to next index interval
                else:
                    indices_used = np.int64([])
                    while num_used > num_group_samples:
                        indices_used = np.concatenate((indices_used, group_indices))  # fill in the remaining data
                        np.random.shuffle(group_indices)
                        num_used -= num_group_samples

                    # fill in the remaining data
                    indices_used = np.concatenate((indices_used, group_indices[:num_used]))
                    # select the index and rearrange
                    indices_used = indices_used.reshape(-1, num_batches)
                    indices.append(indices_used)        # append to the total list
                    indices_start += num_group_samples  # jump to next index interval
            else:
                # mode=2, record the remaining data index, remaining data is prior to sample
                unused_num = index_unused[dataset_count].shape[0]   # record the unused data
                if unused_num >= num_used:  # remaining data is more than the data to be used
                    indices_used = index_unused[dataset_count][:num_used]
                    index_unused[dataset_count] = index_unused[dataset_count][num_used:]
                else:
                    if unused_num > 0:  # remaining data is less than the data to be used
                        indices_used = index_unused[dataset_count]  # use these part of index
                        num_used -= unused_num                      # remaining part of index
                    else:
                        indices_used = np.int64([])
                    group_indices = np.arange(indices_start, indices_start + num_group_samples)  # restart
                    np.random.shuffle(group_indices)  # shuffle the index interval

                    # fill in the remaining data
                    indices_used = np.concatenate((indices_used, group_indices[:num_used]))
                    index_unused[dataset_count] = group_indices[num_used:]  # use the unused data

                # select the index and rearrange
                indices_used = indices_used.reshape(-1, num_batches)
                indices.append(indices_used)        # append to the total list
                indices_start += num_group_samples  # jump to next index interval
                dataset_count += 1
        indices = np.concatenate(indices).transpose().reshape(-1).tolist()  # concatenate

    elif mode == -1:
        if sample_type == "dist":
            # distributed multiple gpu, total_size is the real sample numbers
            group_indices = np.arange(0, total_size)
        else:
            group_indices = np.arange(0, num_samples)
        np.random.shuffle(group_indices)
        indices = group_indices.tolist()

    else:
        raise ValueError('mode ValueError')

    return indices


@SAMPLER.register_module()
class BatchBalancedSampler(Sampler):
    """ Implementation of batched balance sampler"""

    def __init__(self, dataset, mode=0, samples_per_gpu=1, shuffle=True):
        """
            This sampler is used for sampling the batch according to the given batch_ratio
        Args:
            dataset (dataset): dataset for sampling
            mode (int): sampler mode, including [-1, 0, 1, 2]
            samples_per_gpu (int): image numbers in each gpu
            shuffle (bool): whether to shuffle data
        """

        assert hasattr(dataset, 'flag')
        assert mode in [-1, 0, 1, 2]

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag
        self.mode = mode
        self.shuffle = shuffle
        self.sample_type = None
        self.batch_ratios = None
        self.index_unused = None

        assert 'group_samples' in self.flag
        self.group_samples = np.array(self.flag['group_samples']).astype(np.int64)
        if self.mode >= 0:
            assert 'batch_ratios' in self.flag
            self.batch_ratios = np.array(self.flag['batch_ratios']).astype(np.float32)

            # constraint on sum of the batch ratios equals 1
            assert abs(np.sum(self.batch_ratios) - 1) < 0.00001, \
                'sum of the batch ratio should be 1 but got {}'.format(np.sum(self.batch_ratios))

            # constraint batch_size * (sum of batch ratio) equals batch_size
            samples_per_gpu_tmp = sum([round(samples_per_gpu * ratio) for ratio in self.batch_ratios])

            # calculate the difference between calculated the result and user defined batch size
            # according to the calculation accuracy
            num_diff = samples_per_gpu - samples_per_gpu_tmp

            # the difference data are taken from the dataset with the largest batch_ratio
            max_raito_index = np.where(self.batch_ratios == np.max(self.batch_ratios))[0][0]
            tmp_raito = (round(self.batch_ratios[max_raito_index] * samples_per_gpu) + num_diff) / samples_per_gpu
            self.batch_ratios[max_raito_index] = min(1.0, max(0.0, tmp_raito))

            if self.mode == 0 or self.mode == 2:  # The first used data treats as one epoch
                num_batches = math.inf
                for batch_ratio, num_group_samples in zip(self.batch_ratios, self.group_samples):
                    group_samples_per_gpu = int(round(batch_ratio * samples_per_gpu))
                    # calculate number of samples in each batch of each dataset
                    num_batches = min(num_group_samples // group_samples_per_gpu, num_batches)
                    # first used data is synthesized into one epoch
                if self.mode == 2:  # Ensure that unused data must be used up
                    self.index_unused = [np.int64([]) for _ in range(len(self.group_samples))]
                    # initialize the data index which are unused
            elif self.mode == 1:  # calculate the epoch as the last unused data
                num_batches = 0
                for batch_ratio, num_group_samples in zip(self.batch_ratios, self.group_samples):
                    group_samples_per_gpu = int(round(batch_ratio * samples_per_gpu))
                    # calculate sample number in each batch of each dataset
                    num_batches = max(num_group_samples // group_samples_per_gpu, num_batches)
                    # calculate the epoch as the last unused data
            else:
                raise ValueError("Not supported sampler type !!!")

        elif self.mode == -1:
            # shuffle total data, without batch sample operation
            num_samples = 0
            for num_group_samples in self.group_samples:
                num_samples += num_group_samples
            num_batches = num_samples // samples_per_gpu
        else:
            raise ValueError('mode ValueError')

        # sample number of each epoch
        self.num_samples = num_batches * samples_per_gpu
        self.num_batches = num_batches

    def __iter__(self):
        """
        Returns:
            np.array: image sample index

        """
        # initialization each epoch
        indices = batch_sampler(sample_type=self.sample_type, epoch=None, mode=self.mode, num_replicas=1,
                                num_samples=self.num_samples, batch_ratios=self.batch_ratios,
                                group_samples=self.group_samples, samples_per_gpu=self.samples_per_gpu,
                                num_batches=self.num_batches, index_unused=self.index_unused,
                                shuffle=self.shuffle, total_size=None)

        assert len(indices) == self.num_samples, \
            ' indices != num_samples : {} != {}'.format(len(indices), self.num_samples)
        return iter(indices)

    def __len__(self):
        """

        Returns:
            numbers of the sample images

        """
        return self.num_samples


@SAMPLER.register_module()
class DistBatchBalancedSampler(_DistributedSampler):
    """
        This sampler is used for sampling
        the batch according to the given batch_ratio
    """
    def __init__(self, dataset, mode=0,
                 num_replicas=None,
                 rank=None, shuffle=True,
                 samples_per_gpu=1):
        """
        This distributed sampler is used for sampling the batch according to the given batch_ratio
        Args:
            dataset (dataset): dataset for sampling
            mode (int): sampler mode, including [-1, 0, 1, 2]
            num_replicas (int): distributed gpu number
            rank (int): device index
            shuffle (bool): whether to shuffle data
            samples_per_gpu (int): image numbers in each gpu
        """

        assert hasattr(dataset, 'flag')
        assert mode in [-1, 0, 1, 2]

        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag
        self.mode = mode
        self.sample_type = "dist"
        self.index_unused = list()
        self.batch_ratios = None
        self.index_unused = None

        assert 'group_samples' in self.flag
        self.group_samples = np.array(self.flag['group_samples']).astype(np.int64)

        if self.mode >= 0:
            assert 'batch_ratios' in self.flag
            self.batch_ratios = np.array(self.flag['batch_ratios']).astype(np.float32)

            # constraint on sum of the batch ratios equals 1
            assert abs(np.sum(self.batch_ratios) - 1) < 0.00001, \
                'sum of the batch ratio should be 1 but got {}'.format(np.sum(self.batch_ratios))

            # constraint batch_size * (sum of batch ratio) equals batch_size
            samples_per_gpu_tmp = sum([round(samples_per_gpu * ratio) for ratio in self.batch_ratios])

            # calculate the difference between calculated the result and user defined batch size
            # according to the calculation accuracy
            num_diff = samples_per_gpu - samples_per_gpu_tmp

            # the difference data are taken from the dataset with the largest batch_ratio
            max_raito_index = np.where(self.batch_ratios == np.max(self.batch_ratios))[0][0]
            tmp_raito = (round(self.batch_ratios[max_raito_index] * samples_per_gpu) + num_diff) / samples_per_gpu
            self.batch_ratios[max_raito_index] = min(1.0, max(0.0, tmp_raito))

            if self.mode == 0 or self.mode == 2:  # The first used data treats as one epoch
                num_batches = math.inf
                for batch_ratio, num_group_samples in zip(self.batch_ratios, self.group_samples):
                    group_samples_per_gpu = int(round(batch_ratio * samples_per_gpu))
                    # calculate number of samples in each batch of each dataset
                    num_batches = min(num_group_samples // group_samples_per_gpu // self.num_replicas,
                                      num_batches)
                    # first used data is synthesized into one epoch
                if self.mode == 2:  # Ensure that unused data must be used up
                    # initialize the data index which are unused
                    self.index_unused = [np.int64([]) for _ in range(len(self.group_samples))]
            elif self.mode == 1:  # calculate the epoch as the last unused data
                num_batches = 0
                for batch_ratio, num_group_samples in zip(self.batch_ratios, self.group_samples):
                    group_samples_per_gpu = int(round(batch_ratio * samples_per_gpu))
                    # calculate sample number in each batch of each dataset
                    num_batches = max(num_group_samples // group_samples_per_gpu // self.num_replicas,
                                      num_batches)
                    # calculate the epoch as the last unused data
            else:
                raise ValueError("Not supported sampler type !!!")

        elif self.mode == -1:
            # shuffle total data, without batch sample operation
            num_samples = 0
            for num_group_samples in self.group_samples:
                num_samples += num_group_samples
            num_batches = num_samples // samples_per_gpu // self.num_replicas
        else:
            raise ValueError('mode ValueError')

        # sample number of each epoch
        self.num_samples = num_batches * samples_per_gpu
        self.num_batches = num_batches
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        Returns:
            np.array: image sample index

        """
        # initialization each epoch
        indices = batch_sampler(self.sample_type,
                                self.epoch,
                                self.mode,
                                self.num_replicas,
                                self.total_size,
                                self.batch_ratios,
                                self.group_samples,
                                self.samples_per_gpu,
                                self.num_batches,
                                self.index_unused,
                                self.shuffle,
                                self.total_size)

        indices = indices[self.rank:self.total_size:self.num_replicas]

        assert len(indices) == self.num_samples, ' indices != num_samples : {} != {}'.format(len(indices),
                                                                                             self.num_samples)

        return iter(indices)

    def __len__(self):
        """
        Returns:
            int: image sample index

        """
        return self.num_samples

    def set_epoch(self, epoch):
        """
        Args:
            epoch (int): epoch number

        Returns:

        """

        self.epoch = epoch
