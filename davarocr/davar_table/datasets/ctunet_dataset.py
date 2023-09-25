"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ctunet_dataset.py
# Abstract       :    Dataset format used in CTUNet

# Current Version:    1.0.0
# Date           :    2022-11-22
##################################################################################################
"""
import copy
import json
import os
import numpy as np

import torch
from mmdet.datasets.builder import DATASETS

from davarocr.davar_common.datasets import DavarCustomDataset
from davarocr.davar_table.core.evaluation import evaluate_tree_f1


@DATASETS.register_module()
class CTUNetDataset(DavarCustomDataset):
    """ Dataset encapsulation for ctunet dataset from paper <<End-to-End Compound Table Understanding with Multi-Modal
        Modeling>>
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
            if isinstance(per[0], str):
                per_box_label.append(self.CLASSES.index(per[0]))
            else:
                per_box_label.append(per[0])

        return torch.Tensor(per_box_label)

    def get_relations(self, idx):
        """Process data_infos to get bbox_labels.

        Args:
            idx (int): index of sample in data_infos

        Returns:
            Tensor : bbox labels
        """
        box_info = self.data_infos[idx]['ann']

        return box_info['relations']

    def compute_f1_score(self, preds, gts, ignores=None):
        """Compute the F1-score of prediction.

        Args:
            preds (Tensor): The predicted probability NxC map with N and C being the sample number
                and class number respectively.
            gts (Tensor): The ground truth vector of size N.
            ignores (list): The index set of classes that are ignored when reporting results.
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

        return classes, f1_score[torch.LongTensor(classes)].cpu().numpy(), precisions[
            torch.LongTensor(classes)].cpu().numpy(), recalls[torch.LongTensor(classes)].cpu().numpy()

    def save_results(self, results, save_prefix, save_name):
        """Save prediction result to json file.

        Args:
            results (Tensor): The predicted probability NxC map with N and C being the sample number
            save_prefix (string): Storage directory
            save_name (string): Document name.
        """
        res = {}
        for idx, per_result in enumerate(results):
            # get ground-truth of cells' location, texts and row/column numbers.
            filename = self.data_infos[idx]['filename']
            res[filename] = {'content_ann': {}}
            res[filename]['content_ann']['bboxes'] = self.data_infos[idx]['ann']['bboxes']
            res[filename]['content_ann']['texts'] = self.data_infos[idx]['ann']['texts']
            res[filename]['content_ann']['cells'] = self.data_infos[idx]['ann']['cells']

            # cell type classification
            cls_pred_np = np.argmax(per_result['bboxes_labels_pred'], -1)
            labels = [[int(label)] for label in cls_pred_np.ravel()]
            res[filename]['content_ann']['labels'] = labels

            # relation linking of cells
            relations = [[[0] for m in range(len(labels))] for _ in range(len(labels))]
            row_pred_np = np.argmax(results[idx]['bboxes_edges_pred_row'], -1)  # row linking
            col_pred_np = np.argmax(results[idx]['bboxes_edges_pred_col'], -1)  # col linking
            for i in range(len(relations)):
                for j in range(len(relations)):
                    if int(col_pred_np[i, j]):
                        relations[i][j] = 1
                    elif int(row_pred_np[i, j]):
                        relations[i][j] = 2
                    else:
                        relations[i][j] = 0
            res[filename]['content_ann']['relations'] = relations

        with open(os.path.join(save_prefix, save_name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(res, ensure_ascii=False))

        return

    def evaluate(self,
                 results,
                 logger=None,
                 metric='macro_f1',
                 metric_options=None,
                 **eval_kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | None | str): Logger used for printing related information during evaluation.
                Default: None.
            metric (str | list[str]): Metrics to be evaluated. Default: 'macro_f1'.
            metric_options (dict): specify the ignores classes if exist.

        Returns:
            dict: evaluation results.
        """
        eval_results = dict()

        if metric_options is None:
            metric_options = dict(macro_f1=dict(ignores=[]))
        metric_options = copy.deepcopy(metric_options)
        ignores = metric_options['macro_f1'].get('ignores', [])
        save_result = metric_options.get('save_result', False)

        if save_result:
            save_name = metric_options.get('save_name', 'ctunet_result.json')
            save_prefix = metric_options.get('save_prefix', '/data1/save/')
            self.save_results(results, save_prefix=save_prefix, save_name=save_name)
            eval_results['filename'] = save_name
        else:
            if not isinstance(metric, str):
                assert len(metric) == 1
                metric = metric[0]
            allowed_metrics = ['macro_f1', 'hard_f1', 'tree_f1']
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

            # macro f1 metric of cells classification
            annotations = [self.process_anns(i) for i in range(len(self))]
            preds = [per_result['bboxes_labels_pred'] for per_result in results]
            preds = torch.from_numpy(np.concatenate(preds, axis=0))
            gts = torch.cat(annotations).int()
            classes, node_f1s, node_pre, node_recall = self.compute_f1_score(preds, gts, ignores=ignores)
            classes_f1 = node_f1s.tolist()
            print_info = ['{}:{}'.format(name, value) for name, value in zip(classes, classes_f1)]

            eval_results['macro_f1'] = node_f1s.mean().item()
            eval_results['macro_precision'] = node_pre.mean().item()
            eval_results['macro_recall'] = node_recall.mean().item()
            eval_results['classes_f1'] = print_info

            # f1 metric of edge classification
            edge_row_f1, edge_col_f1 = [], []
            relations_list = [self.get_relations(i) for i in range(len(self))]
            relations_pred_list = []
            for i in range(len(self)):
                # row linking
                edge_pred_np = np.argmax(results[i]['bboxes_edges_pred_row'], -1)
                relations_pred = edge_pred_np.copy()
                relations_pred[relations_pred == 1] = 2

                if relations_list[i]:
                    edge_gt_np = np.array(relations_list[i])
                    edge_gt_np[edge_gt_np == 1] = 0
                    edge_gt_np[edge_gt_np == 2] = 1
                else:
                    edge_gt_np = np.zeros_like(edge_pred_np)

                edge_gt_np = edge_gt_np.reshape(-1)
                edge_pred_np = edge_pred_np.reshape(-1)
                if (edge_gt_np > 0).sum() + (edge_pred_np > 0).sum() >= 0.5:
                    edge_correct = ((edge_gt_np == edge_pred_np) & (edge_gt_np > 0)).sum()
                    f1 = 2 * edge_correct / ((edge_gt_np > 0).sum() + (edge_pred_np > 0).sum())
                    edge_row_f1.append(float(f1))

                # col linking
                edge_pred_np = np.argmax(results[i]['bboxes_edges_pred_col'], -1)
                relations_pred[edge_pred_np == 1] = 1
                relations_pred_list.append(relations_pred.astype(int).tolist())

                if relations_list[i]:
                    edge_gt_np = np.array(relations_list[i])
                    edge_gt_np[edge_gt_np == 1] = 1
                    edge_gt_np[edge_gt_np == 2] = 0
                else:
                    edge_gt_np = np.zeros_like(edge_pred_np)

                edge_gt_np = edge_gt_np.reshape(-1)
                edge_pred_np = edge_pred_np.reshape(-1)
                if (edge_gt_np > 0).sum() + (edge_pred_np > 0).sum() > 0.5:
                    edge_correct = ((edge_gt_np == edge_pred_np) & (edge_gt_np > 0)).sum()
                    f1 = 2 * edge_correct / ((edge_gt_np > 0).sum() + (edge_pred_np > 0).sum())
                    edge_col_f1.append(float(f1))

            eval_results['img_avg_row_linking_f1'] = np.array(edge_row_f1).sum() / (len(edge_row_f1))
            eval_results['img_avg_col_linking_f1'] = np.array(edge_col_f1).sum() / (len(edge_col_f1))

            # tree-f1-score used in paper <<End-to-End Compound Table Understanding with Multi-Modal Modeling>>
            hard_recall, hard_precision, hard_f1 = evaluate_tree_f1(relations_pred_list, relations_list,
                                                                    eval_type='hard')
            eval_results['hard_recall'] = hard_recall
            eval_results['hard_precision'] = hard_precision
            eval_results['hard_f1'] = hard_f1
            tree_recall, tree_precision, tree_f1 = evaluate_tree_f1(relations_pred_list, relations_list,
                                                                    eval_type='soft')
            eval_results['tree_recall'] = tree_recall
            eval_results['tree_precision'] = tree_precision
            eval_results['tree_f1'] = tree_f1

        return eval_results
