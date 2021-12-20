"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    wildreceipt_dataset.py
# Abstract       :    Online evaluation of wildreceipt dataset, from https://github.com/open-mmlab/mmocr.

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
import copy
import numpy as np

import torch
from mmdet.datasets.builder import DATASETS

from davarocr.davar_common.datasets import DavarCustomDataset


@DATASETS.register_module()
class WildReceiptDataset(DavarCustomDataset):
    """ Dataset encapsulation for wildreceipt dataset from <<Spatial Dual-Modality Graph Reasoning for Key
        Information Extraction>>
    """
    def __init__(self,
                 **kwargs):
        """Same with DavarCustomDataset."""
        super().__init__(**kwargs)

    def process_anns(self, idx):
        """Process data_infos to get bbox_labels.

        Args:
            idx (int): index of sample in data_infos

        Returns:
            Tensor : bbox labels
        """
        box_info = self.data_infos[idx]['ann']
        per_box_label = []
        for iter_idx, per in enumerate(box_info['labels']):
            if box_info['cares'][iter_idx] != 0:
                if isinstance(per[0], str):
                    per_box_label.append(self.CLASSES.index(per[0]))
                else:
                    per_box_label.append(per[0])

        return torch.Tensor(per_box_label)

    def compute_f1_score(self, preds, gts, ignores=None):
        """Compute the F1-score of prediction.

        Args:
            preds (Tensor): The predicted probability NxC map
                with N and C being the sample number and class
                number respectively.
            gts (Tensor): The ground truth vector of size N.
            ignores (list): The index set of classes that are ignored when
                reporting results.
                Note: all samples are participated in computing.

         Returns:
            List: class ids cared
         Returns:
            np.Array: f1-scores of valid classes.
        """
        if ignores is None:
            ignores = []
        num_classes = preds.size(1)
        classes = sorted(set(range(num_classes)) - set(ignores))
        hist = torch.bincount(
            gts * num_classes + preds.argmax(1), minlength=num_classes ** 2).view(num_classes, num_classes).float()
        diag = torch.diag(hist)
        recalls = diag / hist.sum(1).clamp(min=1)
        precisions = diag / hist.sum(0).clamp(min=1)
        f1_score = 2 * recalls * precisions / (recalls + precisions).clamp(min=1e-8)

        return classes, f1_score[torch.LongTensor(classes)].cpu().numpy()

    def evaluate(self,
                 results,
                 logger=None,
                 metric='macro_f1',
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
        if metric_options is None:
            metric_options = dict(macro_f1=dict(ignores=[]))
        metric_options = copy.deepcopy(metric_options)
        ignores = metric_options['macro_f1'].get('ignores', [])

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['macro_f1']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))
        annotations = [self.process_anns(i) for i in range(len(self))]

        preds = [per_result['bboxes_labels_pred'] for per_result in results]
        preds = torch.from_numpy(np.concatenate(preds, axis=0))

        gts = torch.cat(annotations).int()

        classes, node_f1s = self.compute_f1_score(preds, gts, ignores=ignores)
        classes_f1 = node_f1s.tolist()

        print_info = ['{}:{}'.format(name, value) for name, value in zip(classes, classes_f1)]

        eval_results = dict()
        eval_results['macro_f1'] = node_f1s.mean().item()
        eval_results['classes_f1'] = print_info

        return eval_results
