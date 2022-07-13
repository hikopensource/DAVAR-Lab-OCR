"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_custom.py
# Abstract       :    Implementation of the common dataset of davar group, which supports tasks of
                      Object Detection, Classification, Semantic Segmentation, OCR, etc.

# Current Version:    1.0.0
# Date           :    2020-11-18
##################################################################################################
"""
import copy
import os.path as osp
import collections
import json
from collections import OrderedDict
import numpy as np

import mmcv
from mmcv.utils import print_log

from mmdet.datasets import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from mmdet.core import eval_map, eval_recalls


@DATASETS.register_module()
class DavarCustomDataset(CustomDataset):
    """ Implementation of the common dataset of davar group, which supports tasks of
        Object Detection, Classification, Semantic Segmentation, OCR, etc. Properties in 'content_ann' can be chosen
        according to different tasks.

       train_datalist.json:                                                        # file name
        {
            "###": "Comment",                                                      # The meta comment
            "Images/train/img1.jpg": {                                             # Relative path of images
                "height": 534,                                                     # Image height
                "width": 616,                                                      # Image width
                "content_ann": {                                                   # Following lists have same lengths.
                    "bboxes": [[161, 48, 563, 195, 552, 225, 150, 79],             # Bounding boxes in shape of [2 * N]
                                [177, 178, 247, 203, 240, 224, 169, 198],          # where N >= 2. N=2 means the
                                [263, 189, 477, 267, 467, 296, 252, 218],          # axis-alignedrect bounding box
                                [167, 211, 239, 238, 232, 256, 160, 230],
                                [249, 227, 389, 278, 379, 305, 239, 254],
                                [209, 280, 382, 343, 366, 384, 194, 321]],
                    "cbboxes": [ [[...],[...]], [[...],[...],[...]],               # Character-wised bounding boxes
                    "cares": [1, 1, 1, 1, 1, 0],                                   # If the bboxes will be cared
                    "labels": [['title'], ['code'], ['num'], ['value'], ['other]], # Labels for classification/detection
                                                                                   # task, can be int or string.
                    "texts": ['apple', 'banana', '11', '234', '###'],              # Transcriptions for text recognition
                }
                "content_ann2":{                                                   # Second-level annotations
                    "labels": [[1],[2],[1]]
                }
                "answer_ann":{                                                  # Structure information k-v annotations
                    "keys": ["title", "code", "num","value"],                   # used in end-to-end evaluation
                    "value": [["apple"],["banana"],["11"],["234"]]
                }
            },
            ....
        }
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
                 ):
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
        """

        self.CLASSES = self.get_classes(classes)
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        # Join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)

        # Load annotations (and proposals)
        data_infos = self.load_annotations(self.ann_file)
        self.data_infos = self._cvt_list(data_infos)
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # Filter images with no annotation during training
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # Set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # Processing pipeline
        self.pipeline = Compose(pipeline)
        if classes_config is not None:
            self.classes_config = mmcv.load(classes_config)
        else:
            self.classes_config = None

    def _load_json(self, json_file):
        """ Load annotations for JSON file

        Args:
            json_file(str): the path to JSON file

        Returns:
            dict: annotation information obj
        """
        with open(json_file, 'r', encoding='utf-8') as load_f:
            ann = json.load(load_f, object_pairs_hook=collections.OrderedDict)
        return ann

    def _cvt_list(self, img_info):
        """ Convert JSON dict into a list.

        Args:
            img_info(dict): annotation information in a json obj

        Returns:
            list(dict): converted list of annotations, in form of
                       [{"filename": "xxx", width: 120, height: 320, ann: {}, ann2: {}},...]
        """

        result_dict = []

        # Remove the meta comment in json
        if "###" in img_info.keys():
            del img_info["###"]

        for key in img_info.keys():
            tmp_dict = dict()
            tmp_dict["filename"] = key
            tmp_dict["height"] = img_info[key]["height"]
            tmp_dict["width"] = img_info[key]["width"]
            tmp_dict["ann"] = img_info[key]["content_ann"]
            tmp_dict["ann2"] = img_info[key].get("content_ann2", None)
            result_dict.append(tmp_dict)

        return result_dict

    def _filter_imgs(self, min_size=32):
        """Filter images too small and images without annoations

        Args:
            min_size(in): minimum supported image size

        Returns:
            list(int): the valid indexes of images.
        """

        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            # Not support gif
            if img_info['filename'].split(".")[-1].upper() == 'GIF':
                continue

            ann = img_info['ann']

            # Filter images with empty ground-truth
            if ann is not None and self.filter_empty_gt:
                if ('bboxes' in ann and len(ann['bboxes']) == 0) or ('cares' in ann and 1 not in ann['cares']):
                    continue

            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def load_annotations(self, ann_file):
        """ Load annotation from file

        Args:
            ann_file(str): path to annotation file

        Returns:
            dict: loaded annotation obj.
        """
        return self._load_json(ann_file)

    def load_proposals(self, proposal_file):
        """ Load proposals from file

        Args:
            proposal_file(str): path to proposal file

        Returns:
            dict: loaded proposal obj.
        """
        return self._load_json(proposal_file)

    def get_ann_info(self, idx):
        """ Get the annotation of the specific instance, corresponding to `content_ann`.

        Args:
            idx(int): instance id.

        Returns:
            dict: the annotation of the instance.
        """
        return self.data_infos[idx].get('ann', None)

    def get_ann_info_2(self, idx):
        """ Get the 2-nd level annotation of the specific instance, corresponding to `content_ann2` .

        Args:
            idx(int): instance id.

        Returns:
            dict: the annotation of the instance.
        """
        return self.data_infos[idx].get('ann2', None)

    def pre_pipeline(self, results):
        """ Prepare pipelines. Integrated with some preset keys, like `bbox_fields`, `cbbox_fields`

        Args:
            results(dict): original data flow

        Returns:
            dict: updated data flow
        """
        super().pre_pipeline(results)
        results['cbbox_fields'] = []

    def prepare_train_img(self, idx):
        """ Prepare training data annotation and send into pipelines

        Args:
            idx(int): the instance id

        Returns:
            dict: the formated data that was prepared for training.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        ann_info_2 = self.get_ann_info_2(idx)
        results = dict(img_info=img_info, ann_info=ann_info, ann_info_2=ann_info_2, classes_config=self.classes_config)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """ Prepare testing data annotation and send into pipelines

        Args:
            idx(int): the instance id

        Returns:
            dict: the formated data that was prepared for testing.
        """
        return self.prepare_train_img(idx)

    def process_anns(self, idx):
        """ process dataset format for evaluation

        Args:
            idx(int): instance index

        Returns:
            dict: formatted data info
        """
        img_info = copy.deepcopy(self.data_infos[idx].get('ann', None))
        if self.classes_config is not None:
            img_info['labels'] = [per[0] for per in img_info['labels']]
            bboxes = []
            labels = []
            bboxes_ignore = []
            labels_ignore = []
            cares = img_info.get('cares', None)
            if cares is None:
                cares = [1] * len(img_info['labels'])

            for i, care in enumerate(cares):
                x_min = min(img_info['bboxes'][i][0::2])
                x_max = max(img_info['bboxes'][i][0::2])
                y_min = min(img_info['bboxes'][i][1::2])
                y_max = max(img_info['bboxes'][i][1::2])
                rect_box = [x_min, y_min, x_max, y_max]
                if care:
                    bboxes.append(rect_box)
                    labels.append(self.classes_config['classes'].index(img_info['labels'][i]))
                else:
                    bboxes_ignore.append(rect_box)
                    labels_ignore.append(self.classes_config['classes'].index(img_info['labels'][i]))
            bboxes = np.array(bboxes).reshape(-1, 4)
            bboxes_ignore = np.array(bboxes_ignore).reshape(-1, 4)
            labels = np.array(labels)
            labels_ignore = np.array(labels_ignore)
            img_info['bboxes'] = bboxes
            img_info['bboxes_ignore'] = bboxes_ignore
            img_info['labels'] = labels
            img_info['labels_ignore'] = labels_ignore
        return img_info

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 **eval_kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            eval_kwargs (dict): other eval kwargs.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if len(results) > 0 and isinstance(results[0], dict):
            num_classes = len(self.classes_config['classes'])
            tmp_results = []
            for res in results:
                points = np.array(res['points']).reshape(-1, 4)
                scores = np.array(res['scores']).reshape(-1, 1)
                labels = np.array(res['labels'])
                bboxes = np.concatenate([points, scores], axis=-1)
                tmp_results.append([bboxes[labels == i, :] for i in range(num_classes)])
            results = tmp_results
        annotations = [self.process_anns(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for tmp_iou_thre in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {tmp_iou_thre}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=tmp_iou_thre,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(tmp_iou_thre * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                arr = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = arr[i]
        return eval_results
