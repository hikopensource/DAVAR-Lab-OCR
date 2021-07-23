"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_multi_dataset.py
# Abstract       :    Implementation of the multiple dataset loading of davar group.

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""

import bisect

from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

from mmdet.datasets import DATASETS


@DATASETS.register_module()
class DavarMultiDataset(Dataset):
    """MultiDataset: Support for different sample ratios from different dataset"""
    CLASSES = None

    def __init__(self,
                 batch_ratios,
                 dataset,
                 test_mode=False):
        """
        davar multiple dataset loading

        Args:
            batch_ratios (int|list|str): use ratio on each dataset in each batch
            dataset (dataset): dataset for concatenation
        """
        # parameter initialization
        self.test_mode = test_mode

        if isinstance(batch_ratios, (float, int)):
            batch_ratios = [batch_ratios]
        elif isinstance(batch_ratios, (tuple, list)):
            pass
        else:
            batch_ratios = list(map(float, batch_ratios.split('|')))
        self.batch_ratios = batch_ratios

        self.datasets = list()

        for _, dataset_ in enumerate(dataset):
            print('number of samples:', len(dataset_))
            self.datasets.append(dataset_)

        # concat all the dataset
        self.concated_dataset = ConcatDataset(self.datasets)

        if not self.test_mode:
            self._set_group_flag()

    def __len__(self):
        """

        Returns:
            int: length of the dataset

        """

        return len(self.concated_dataset)

    def _set_group_flag(self):
        """
        capture the parameter in config
        """

        group_samples = list()

        for dataset in self.datasets:
            group_samples.append(len(dataset))

        self.flag = dict()
        self.flag['batch_ratios'] = self.batch_ratios
        self.flag['group_samples'] = group_samples

    def prepare_train_img(self, idx):
        """
        prepare for the train image
        Args:
            idx (int): training sample index

        Returns:
            Dataset: training sample index corresponding image sample
        """

        data = self.concated_dataset.__getitem__(idx)
        return data

    def prepare_test_img(self, idx):
        """
        prepare for the test image
        Args:
            idx (int): test sample index

        Returns:
            np.array: test sample index corresponding dataset

        """
        data = self.concated_dataset.__getitem__(idx)
        return data

    def __getitem__(self, idx):
        """
        Args:
            idx (int): sample index

        Returns:
             np.array: sample index corresponding image sample

        """
        assert idx < len(self), 'index range error'
        if self.test_mode:
            return self.prepare_test_img(idx)
        return self.prepare_train_img(idx)

    def get_ann_info(self, idx):
        """
            get training label information
        Args:
            idx (int): sample index

        Returns:
            text: sample index corresponding label information

        """
        dataset_idx = bisect.bisect_right(self.concated_dataset.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.concated_dataset.cumulative_sizes[dataset_idx - 1]
        return self.concated_dataset.datasets[dataset_idx].get_ann_info(sample_idx)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 logger=None,
                 **eval_kwargs):
        """
             model evaluation api
        Args:
            results (list): model prediction results
            metric (str): evaluation metric
            logger (logging.Logger): Logger used for printing related information during evaluation. Default: None.

        Returns:
            dict: model evaluation metric

        """
        validation_result = self.datasets[0].evaluate(results, metric, logger, **eval_kwargs)
        return validation_result
