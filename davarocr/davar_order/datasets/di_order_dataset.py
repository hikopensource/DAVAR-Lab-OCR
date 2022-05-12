"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    sentence_order_dataset.py
# Abstract       :    Online evaluation of DI dataset.

# Current Version:    1.0.0
# Date           :    2022-05-12
##################################################################################################
"""
from mmdet.datasets.builder import DATASETS
from davarocr.davar_common.datasets import DavarCustomDataset


@DATASETS.register_module()
class DIOrderDataset(DavarCustomDataset):
    """ Dataset encapsulation for DI dataset from <<An End-to-End OCR Text Re-organization Sequence Learning
        for Rich-text Detail Image Comprehension>>
    """
    def __init__(self,
                 **kwargs):
        """Same with DavarCustomDataset."""
        super().__init__(**kwargs)

    def get_orders(self, idx):
        """Process data_infos to get bbox_labels.

        Args:
            idx (int): index of sample in data_infos

        Returns:
            Tensor : bbox labels
        """
        box_info = self.data_infos[idx]['ann']

        return box_info['labels']

    def evaluate(self,
                 results,
                 logger=None,
                 metric='total_order_acc',
                 metric_options=None,
                 **eval_kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            metric_options (dict): specify the ignores classes if exist.

        Returns:
            dict: evaluation results.
        """
        eval_results = dict()
        assert len(results) == len(self.data_infos)
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['total_order_acc']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        gt_labels = []
        for i in range(len(results)):
            labels = self.get_orders(i)
            order = [label[0]-1 for label in labels]
            gt_labels.append(order)
        cor = 0
        for gt, pred in zip(gt_labels, results):
            if gt == pred:
                cor += 1
        eval_results['total_order_acc'] = cor/len(results)
        return eval_results
