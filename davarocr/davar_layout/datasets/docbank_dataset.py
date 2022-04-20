"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    docbank_dataset.py
# Abstract       :    Dataset definition for docbank dataset.

# Current Version:    1.0.0
# Date           :    2020-12-06
##################################################################################################
"""
import json
import os
import copy
import random
import torch
import numpy as np

from mmdet.models.losses import accuracy
from mmdet.datasets.builder import DATASETS
from .mm_layout_dataset import MMLayoutDataset


@DATASETS.register_module()
class DocBankDataset(MMLayoutDataset):
    """
    Dataset defination for DocBank dataset.

    Ref: [1] DocBank: A Benchmark Dataset for Document Layout Analysis, COLING 2020.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 classes_config=None,
                 classes=None,
                 ann_prefix='',
                 eval_level=0,
                 max_num=1024):
        """
        Args:
            ann_file(str): the path to datalist.
            pipeline(list(dict)): the data-flow handling pipeline
            data_root(str): the root path of the dataset
            img_prefix(str): the image prefixes
            seg_prefix(str): the segmentation maps prefixes
            proposal_file(str): the path to the preset proposal files.
            test_mode(boolean): whether in test mode
            filter_empty_gt(boolean): whether to filter out image without ground-truthes.
            classes_config(str): the path to classes config file, used to transfer 'str' labels into 'int'
            classes(str): Dataset class, default None.
            ann_prefix(str): Annotation prefix path for each annotation file.
            eval_level(int): evaluation in which level. 1 for highest level, 0 for lowest level.
            max_num(int): specify the max number of tokens loading.
        """
        self.max_num = max_num
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            classes_config=classes_config,
            ann_prefix=ann_prefix,
            classes=classes,
            eval_level=eval_level
        )

    def pre_prepare(self, img_info):
        """Load per annotation file and reset img_info ann& ann2 fields. ann denotes the annotations in token level and
        ann2 in layout level.

        Args:
            img_info(dict): img_info dict.

        Returns:
            dict: updated img_info.

        """
        if img_info['url'] is not None:
            tmp_img_info = copy.deepcopy(img_info)
            ann = json.load(open(os.path.join(self.ann_prefix, tmp_img_info['url']), 'r', encoding='utf8'))

            if "content_ann" in ann.keys():
                tmp_img_info["ann"] = ann["content_ann"]
                cares = ann["content_ann"]["cares"]
                bboxes = ann["content_ann"]["bboxes"]
                cnt_bboxes = 0
                areas = []
                for idx, per_bbox in enumerate(bboxes):
                    w_s, h_s, w_e, h_e = per_bbox
                    area = (w_e - w_s) * (h_e - h_s)
                    areas.append(area)
                    if w_e > w_s and h_e > h_s:
                        cnt_bboxes += 1
                        continue
                    else:
                        # filter bboxes whose area equals 0.
                        cares[idx] = 0

                # we divide all tokens into three groups according to their areas, and sample due to memory limit.
                if cnt_bboxes > self.max_num:
                    area1 = []
                    area10 = []
                    area10_up = []
                    for idx, area in enumerate(areas):
                        if area > 10:
                            area10_up.append(idx)
                        elif area > 1:
                            area10.append(idx)
                        elif area == 1:
                            area1.append(idx)
                        else:
                            continue
                    if len(area1) > self.max_num//16:
                        index = random.sample(area1, len(area1) - self.max_num//16)
                        for i in index:
                            cares[i] = 0

                    if len(area10) > self.max_num//16:
                        index10 = random.sample(area10, len(area10) - self.max_num//16)
                        for i in index10:
                            cares[i] = 0

                    num_res = self.max_num - min(self.max_num//16, len(area1)) - min(self.max_num//16, len(area10))
                    if len(area10_up) > num_res:
                        index10_up = random.sample(area10_up, len(area10_up) - num_res)
                        for i in index10_up:
                            cares[i] = 0

                tmp_img_info["ann"]["cares"] = cares
            else:
                tmp_img_info["ann"] = None

            if "content_ann2" in ann.keys():
                tmp_img_info["ann2"] = ann["content_ann2"]

                # filter wrong labels to not care
                cares = ann["content_ann2"]["cares"]
                bboxes = ann["content_ann2"]["bboxes"]
                for idx, per_bbox in enumerate(bboxes):
                    w_s, h_s, w_e, h_e = per_bbox
                    if w_e > w_s and h_e > h_s:
                        continue
                    else:
                        cares[idx] = 0
                tmp_img_info["ann2"]["cares"] = cares

            else:
                tmp_img_info["ann2"] = None

            return tmp_img_info

        else:
            return img_info

    def evaluate(self,
                 results,
                 logger=None,
                 metric='F1-score'):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        """
        results = [per_r.cpu().numpy()[:, 4:] for per_r in results]
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['acc', 'F1-score']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        annotations = [self.process_anns(i) for i in range(len(self))]
        bboxes = [annotations[i]["bboxes"] for i in range(len(annotations))]
        labels = [annotations[i]["labels"] for i in range(len(annotations))]
        cares = [annotations[i]["cares"] for i in range(len(annotations))]
        labels_care = []
        bboxes_care = []
        classes = self.classes_config["classes_0"]

        # remove not care tokens
        for i in range(len(labels)):
            labels_tmp = [classes.index(labels[i][j]) for j in range(len(labels[i])) if cares[i][j] != 0]
            bboxes_tmp = [bboxes[i][j] for j in range(len(bboxes[i])) if cares[i][j] != 0]
            for j in range(len(labels_tmp)):
                labels_care.append(labels_tmp[j])
                bboxes_care.append(bboxes_tmp[j])

        eval_results = {}
        results = np.array([per_result[j] for per_result in results for j in range(len(per_result))])

        # acc for each category
        if metric == 'acc':
            results_new = [[] for i in range(len(classes))]
            labels_new = [[] for i in range(len(classes))]
            for i in range(len(labels_care)):
                labels_new[labels_care[i]].append(labels_care[i])
                results_new[labels_care[i]].append(results[i])
            for i in range(len(labels_new)):
                results_per = torch.Tensor(np.array(results_new[i]))
                labels_per = torch.Tensor(np.array(labels_new[i]))
                acc_per = accuracy(results_per, labels_per)
                eval_results['acc@{}'.format(classes[i])] = float(acc_per)

        # f1-score for each category
        # calculate pre, recall and f1 according to [1]
        if metric == 'F1-score':
            gt_area, cor_area, pre_area, precision, recall, f1_score = [[0 for i in range(len(classes))] for j in range(6)]
            for i in range(len(labels_care)):
                area = (bboxes_care[i][2]-bboxes_care[i][0])*(bboxes_care[i][3]-bboxes_care[i][1])
                gt_area[labels_care[i]] += area
                label_pre = np.argmax(results[i])
                pre_area[label_pre] += area
                if label_pre == labels_care[i]:
                    cor_area[labels_care[i]] += area
                else:
                    continue
            f1_list = []
            for i in range(len(gt_area)):
                if gt_area[i] == 0:
                    continue
                else:
                    precision[i] = cor_area[i] / (pre_area[i] + 0.01)
                    recall[i] = cor_area[i] / gt_area[i]
                    f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 0.01)
                    eval_results['precision@{}'.format(classes[i])] = float(precision[i])
                    eval_results['recall@{}'.format(classes[i])] = float(recall[i])
                    eval_results['F1 score@{}'.format(classes[i])] = float(f1_score[i])
                    f1_list.append(float(f1_score[i]))

            avg_f1 = sum(f1_list) / (len(f1_list) + 1e-3)
            eval_results['avg_f1'] = avg_f1

        return eval_results
