"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    mm_layout_dataset.py
# Abstract       :    Multi-modal layout analysis base dataset.

# Current Version:    1.0.0
# Date           :    2020-12-06
##################################################################################################
"""
import json
import copy
import os

import numpy as np

from mmdet.datasets.builder import DATASETS
from davarocr.davar_common.datasets import DavarCustomDataset


@DATASETS.register_module()
class MMLayoutDataset(DavarCustomDataset):
    """Multimodal Layout Analysis Dataset definition.

    Since VSR needs annotations of both text line and layout granularities, and PubLayNet and DocBank datasets are too
    large to load in single json file, we define this class and load annotation files for per image.

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
                    "bboxes": [[x0, y0, x1, y1],                           # boxes in layout granularity
                                [x0, y0, x1, y1],                          # in contrast to content_ann, which
                                [x0, y0, x1, y1],                          # is in text/ token granularity.
                                ],                                         # x0, y0 are the coordinates of top-left
                                                                           # corner and x1, y1 the bottom-right corner.
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
                 ann_prefix='',
                 eval_level=1):
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
        """
        self.ann_prefix = ann_prefix
        self.eval_level = eval_level
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
            classes=classes)

    def _cvt_list(self, img_info):
        """ Convert JSON dict into a list.

        Args:
            img_info(dict): annotation information in a json obj

        Returns:
            list(dict): converted list of annotations, in form of
                       [{"filename": "xxx", width: 120, height: 320, ann: {}, ann2: {}},...]
        """
        result_dict = []
        for key in img_info.keys():
            tmp_dict = {}
            tmp_dict["filename"] = key
            tmp_dict["height"] = img_info[key]["height"]
            tmp_dict["width"] = img_info[key]["width"]

            # url
            tmp_dict["url"] = img_info[key].get("url", None)

            # content_ann1
            tmp_dict["ann"] = img_info[key].get("content_ann", None)

            # content_ann2
            tmp_dict["ann2"] = img_info[key].get("content_ann2", None)
            result_dict.append(tmp_dict)

        return result_dict

    def pre_prepare(self, img_info):
        """Load per annotation file and reset img_info ann& ann2 fields.

        Args:
        	img_info(dict): img_info dict.

        Returns:
        	dict: updated img_info.

        """
        if img_info['url'] is not None:
            tmp_img_info = copy.deepcopy(img_info)
            ann = json.load(open(os.path.join(self.ann_prefix, tmp_img_info['url']), 'r', encoding='utf8'))

            tmp_img_info["ann"] = ann.get("content_ann", None)
            if "content_ann2" in ann.keys():
                tmp_img_info["ann2"] = ann.get("content_ann2", None)

                # filter invalid annotations.
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

    def prepare_train_img(self, idx):
        """ Prepare training data annotation and send into pipelines

        Args:
            idx(int): the instance id

        Returns:
            dict: the formated data that was prepared for training.
        """
        img_info = self.data_infos[idx]
        img_info = self.pre_prepare(img_info)

        ann_info = img_info.get('ann', None)
        ann_info_2 = img_info.get('ann2', None)
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
        img_info = self.data_infos[idx]
        img_info = self.pre_prepare(img_info)

        ann_info = img_info.get('ann', None)
        ann_info_2 = img_info.get('ann2', None)

        # return last/ highest level
        if self.eval_level == 1:
            labels = [per[0] for per in ann_info_2['labels']]
            ann_info_2['labels'] = np.array(labels)
            ann_info_2['bboxes'] = np.array(ann_info_2['bboxes'])
            return ann_info_2
        else:
            labels = [per[0] for per in ann_info['labels']]
            ann_info['labels'] = np.array(labels)
            ann_info['bboxes'] = np.array(ann_info['bboxes'])
            return ann_info
