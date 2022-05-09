"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ner_dataset.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import copy
from mmdet.datasets.builder import DATASETS
from ..core.evaluation.ner_metric import eval_ner_f1
from davarocr.davar_nlp_common.datasets import BaseNLPDataset


@DATASETS.register_module()
class NERDataset(BaseNLPDataset):
    """ Custom dataset for named entity recognition tasks.
    """
    def __init__(self, with_multi_labels=False, **kwargs):
        """
        Args:
            with_multi_labels (bool): whether gt contains multi labels.
        """
        super().__init__(**kwargs)
        self.with_multi_labels = with_multi_labels

    def prepare_train_img(self, index):
        """Get training data and annotations after pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        ann_info = self.data_infos[index]
        self.pre_pipeline(ann_info)
        return self.pipeline(ann_info)

    def evaluate(self, results, metric=None, logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: A dict containing the following keys:
             'precision', 'recall', 'f1-score'.
        """
        gt_infos = list(self.data_infos)
        if self.with_multi_labels:
            gt_infos_copy = copy.deepcopy(gt_infos)
            for i, item in enumerate(gt_infos_copy):
                gt_infos[i]['token_labels'] = [label[0] for label in item['token_labels']]
        eval_results = eval_ner_f1(results, gt_infos)
        return eval_results
