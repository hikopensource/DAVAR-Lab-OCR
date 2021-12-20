"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    dataset_wrapper.py
# Abstract       :    Implementation of the concat dataset loading of davar group.

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""

import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from mmdet.datasets import DATASETS


@DATASETS.register_module()
class DavarConcatDataset(_ConcatDataset):
    """ Customized concat dataset, support for different sample ratios for different dataset.
    """

    def __init__(self, datasets):
        """
        A wrapper of concatenated dataset. Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but concat the
        group flag for image aspect ratio.

        Args:
            datasets (dataset): dataset for concatenation
        """

        super().__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.dataset = datasets
        if hasattr(datasets[0], 'flag'):
            if isinstance(datasets[0].flag, dict):
                flags = {
                    'batch_ratios': [],
                    'group_samples': []}
                for i, _ in enumerate(datasets):
                    # training process related the parameter
                    flags['batch_ratios'].extend(datasets[i].flag['batch_ratios'])
                    flags['group_samples'].extend(datasets[i].flag['group_samples'])
                self.flag = flags
            else:
                flags = list()
                for i, _ in enumerate(datasets):
                    flags.append(datasets[i].flag)
                self.flag = np.concatenate(flags)

    def evaluate(self, results, metric='accuracy', logger=None, **eval_kwargs):
        """
        Args:
            results (list): model prediction results
            metric (str): evaluation metric
            logger (logging.Logger): Logger used for printing related information during evaluation. Default: None.

        Returns:
            dict: model evaluation metric

        """

        validation_result = self.datasets[0].evaluate(results=results,
                                                      metric=metric,
                                                      logger=logger,
                                                      **eval_kwargs)
        return validation_result
