"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    base_nlp_dataset.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

from .builder import build_loader


@DATASETS.register_module()
class BaseNLPDataset(Dataset):
    """ Base NLP dataset
    """
    def __init__(self,
                 ann_file,
                 loader,
                 pipeline,
                 img_prefix='',
                 test_mode=False):
        """
        Args:
            ann_file(list or str): annotation files
            loader(dict): loader param
            pipeline(list): transforms pipeline
            img_prefix(str): file's prefix
            test_mode(bool): whether is test mode
        """
        super().__init__()
        self.test_mode = test_mode
        self.img_prefix = img_prefix
        self.ann_file = ann_file

        # load annotations
        loader.update(ann_file=ann_file)
        self.data_infos = build_loader(loader)
        # processing pipeline
        self.pipeline = Compose(pipeline)
        # set group flag and class, no meaning
        # for text detect and recognize
        self._set_group_flag()
        self.CLASSES = 0

    def __len__(self):
        return len(self.data_infos)

    def _set_group_flag(self):
        """Set flag."""
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        #results['img_prefix'] = self.img_prefix
        pass

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.data_infos[index]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, img_info):
        """Get testing data from pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """
        return self.prepare_train_img(img_info)

    def _log_error_index(self, index):
        """Logging data info of bad index."""
        data_info = self.data_infos[index]
        img_prefix = self.img_prefix
        print_log(f'Warning: skip broken file {data_info} '
                      f'with img_prefix {img_prefix}')

    def _get_next_index(self, index):
        """Get next index from dataset."""
        self._log_error_index(index)
        index = (index + 1) % len(self)
        return index

    def __getitem__(self, index):
        """Get training/test data from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training/test data.
        """
        if self.test_mode:
            return self.prepare_test_img(index)

        while True:
            data = self.prepare_train_img(index)
            break
        return data

    def format_results(self, results, **kwargs):
        """Placeholder to format result to dataset-specific output."""
        raise NotImplementedError

    def evaluate(self, results, metric=None, logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        raise NotImplementedError
