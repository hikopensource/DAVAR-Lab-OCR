"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    yoro_rcg_dataset.py
# Abstract       :    Implementation of yoro recognition dataset.

# Current Version:    1.0.0
# Date           :    2021-06-28
##################################################################################################
"""


import os.path as osp

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose

from davarocr.davar_rcg.datasets import DavarRCGDataset
from davarocr.davar_rcg.datasets.pipelines import RcgExtraAugmentation
from davarocr.davar_videotext.tools.test_utils import filter_punctuation


@DATASETS.register_module()
class YORORCGDataset(DavarRCGDataset):
    """ YORO Text Recommender dataset loading """
    CLASSES = None

    def __init__(self,
                 img_prefix,
                 ann_file,
                 data_type,
                 pipeline,
                 batch_max_length=25,
                 used_ratio=1.0,
                 extra_aug=None,
                 test_mode=False,
                 filter_cares=True,
                 filter_scores=False,
                 test_filter=None,
                 not_care_tans='555',
                 filter_punc=True
                 ):
        """
        Args:
            img_prefix (str): the prefix of the dataset annotation file path
            ann_file (str): path to test data list
            data_type (str): type of data loading, Only support 'File'
            pipeline (dict): pre-process pipeline dict
            batch_max_length (int): max recognition text length
            used_ratio (int): used ratio of the total dataset
            extra_aug (dict): extra augmentation dict
            test_mode (bool): whether to be train mode or test mode, Default(False)
            filter_cares(bool): whether to filter the sample which care is 0
            filter_scores(bool): whether to filter the sample which score is none
            test_filter (int): filter necessary information
        """
        # parameter initialization
        self.root = None
        self.filter_cares = filter_cares
        self.filter_scores = filter_scores

        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.batch_max_length = batch_max_length
        self.used_ratio = used_ratio
        self.filtered_index_list = list()
        self.key_list = list()
        self.num_samples = 0
        self.data_type = data_type
        self.not_care_tans = not_care_tans
        self.filter_punc = filter_punc

        if not self.filter_cares:
            print("filter cares is False, do not filter those care field is 0 samples")

        if not self.filter_scores:
            print("filter score is False, do not filter those score field is None samples")

        # Support datalist use absolute path and relative path
        if osp.isfile(osp.join(img_prefix, ann_file)):

            # Datalist use relative path
            self.root = osp.join(img_prefix, ann_file)
        elif osp.isfile(ann_file):

            # Datalist use absolute path
            self.root = ann_file
        else:
            raise Exception("Data file error")

        if test_mode:
            self.phase = "Test"
        else:
            self.phase = "Train"

        self.test_filter = test_filter

        self.filter = filter

        # Build the pipeline
        self.pipeline_dict = pipeline
        self.pipeline = Compose(self.pipeline_dict)

        assert data_type in ['File'], \
            'data_type should be File, but found ' + data_type

        # Extra augmentation operation
        if extra_aug is not None:
            self.extra_aug = RcgExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        self.img_infos = list()

        # File data load
        self.json_file_load(self.root)

    def json_file_load(self, root):
        """

        Args:
            root (str): root path of the File dataset

        Returns:

        """
        # load the label information
        self.img_infos = self._load_annotations(root)

        # load the File data type
        if self.data_type == "File":
            # load the 'box', 'text', 'label' in label information
            self.img_infos = self._file_to_list(self.img_infos)

        self.num_samples = len(self.img_infos)
        # filter the labels whose length are greater than maxlength
        for index in range(self.num_samples):
            label = self.img_infos[index]["ann"]["text"]
            if self.phase == 'Train' and len(label) > self.batch_max_length:
                continue
            self.filtered_index_list.append(index)
            if len(self.filtered_index_list) == int(self.num_samples * self.used_ratio):
                break

        self.num_samples = len(self.filtered_index_list)

    def _file_to_list(self, img_infos):
        """
        Args:
            img_infos (dict): the dict with the label information

        Returns:
            list(dict): select the training sample list

        """
        res_list = list()

        for k, value in img_infos.items():

            # Get preset key-value
            bboxes = value['content_ann']['bboxes']
            texts = value['content_ann'].get('texts', [self.not_care_tans]*len(bboxes))
            labels = value['content_ann'].get('labels', [None]*len(texts))
            tracks = value['content_ann'].get('trackID', [None]*len(texts))
            cares = value['content_ann'].get('cares', [1]*len(texts))
            quality = value['content_ann'].get('qualities', ['HIGH'] * len(texts))
            scores = value['content_ann'].get('score', [None] * len(texts))

            for i, text in enumerate(texts):

                # Filter out defective boxes and polygonal boxes
                if len(bboxes[i]) != 8:
                    continue

                # If filter_cares is True, filter out care[i] != 1 samples, usually for recognition and track task
                if self.filter_cares and not cares[i]:
                    continue

                # If filter_scores is True, filter out scores[i] is None samples, usually for quality score task
                if self.filter_scores and scores[i] is None:
                    continue

                # If filter_punc is True, filter out the punctuation in text
                if self.filter_punc:
                    text = filter_punctuation(text)

                res_list.append({
                    'filename': k,
                    'ann': {
                        'text': text.lower() if not self.pipeline_dict[0]["sensitive"] else text,
                        'bbox': bboxes[i],
                        'label': labels[i],
                        'trackID': tracks[i],
                        'care': cares[i],
                        'quality': quality[i],
                        'score': scores[i]
                    }
                                })
        return res_list
