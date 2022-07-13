"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_rcg_dataset.py
# Abstract       :    Implementations of davar dataset loading

# Current Version:    1.0.1
# Date           :    2022-04-28
##################################################################################################
"""
import os.path as osp
import json
import random
import copy

import lmdb
import mmcv

from torch.utils.data import Dataset
from nltk.metrics.distance import edit_distance

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose

from .pipelines import RcgExtraAugmentation
from ..tools.test_utils import filter_punctuation


@DATASETS.register_module()
class DavarRCGDataset(Dataset):
    """ Davar text Recognition dataset loading """
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
                 test_filter=None,
                 ):
        """
        Args:
            data_type (str): type of data loading, including ["Tight", "LMDB_Davar", "Loose", "LMDBOLD", "File"]
            img_prefix (str): the prefix of the dataset annotation file path
            pipeline (dict): pre-process pipeline dict
            batch_max_length (int): max recognition text length
            used_ratio (int): used ratio of the total dataset
            extra_aug (dict): extra augmentation dict
            test_mode (bool): whether to be train mode or test mode, Default(False)
            test_filter (int): filter necessary information
        """

        # parameter initialization
        self.root = None

        # Process LMDB-type dataset separately
        if "LMDB" in img_prefix:
            self.root = osp.join(img_prefix, ann_file)
        else:
            # Support datalist use absolute path and relative path
            if osp.isfile(osp.join(img_prefix, ann_file)):
                # datalist use relative path
                self.root = osp.join(img_prefix, ann_file)
            elif osp.isfile(ann_file):
                # datalist use absolute path
                self.root = ann_file
            else:
                raise Exception("Data file error")

        self.img_prefix = img_prefix
        self.batch_max_length = batch_max_length
        self.used_ratio = used_ratio
        self.filtered_index_list = list()
        self.key_list = list()
        self.num_samples = 0
        self.env = None
        self.data_type = data_type

        if test_mode:
            self.phase = "Test"
        else:
            self.phase = "Train"

        self.test_filter = test_filter

        self.filter = filter

        # build the pipeline
        self.pipeline_dict = pipeline
        self.pipeline = Compose(self.pipeline_dict)

        assert data_type in ['LMDB_Standard', 'LMDB_Davar', 'File', 'Tight', 'Loose'], \
            'data_type should be LMDB_Standard / LMDB_Davar / File / Tight / Loose, but found ' + data_type

        # extra augmentation operation
        if extra_aug is not None:
            self.extra_aug = RcgExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        self.img_infos = list()
        if "LMDB" in self.data_type:
            # LMDB-type data load
            self.lmdb_file_load(self.root)
        else:
            # File, Tight, Loose-type data load
            self.json_file_load(self.root)

    def lmdb_file_load(self, root):
        """
        Args:
            root (str):
                 the root path of the LMDB dataset

        Returns:

        """

        # LMDB-type data load
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

        if not self.env:
            raise ValueError('cannot create lmdb from %s' % root)

        if 'LMDB' in self.data_type:
            with self.env.begin(write=False) as txn:

                # open-source LMDB data
                if self.data_type == 'LMDB_Standard':

                    # get the total number of the dataset
                    self.num_samples = int(txn.get('num-samples'.encode()))

                    # load the label information, filter label according to the maxlength
                    for index in range(self.num_samples):
                        index += 1  # lmdb starts with 1
                        label_key = 'label-%09d'.encode() % index
                        label = txn.get(label_key).decode('utf-8')
                        if not self.pipeline_dict[0]["sensitive"]:
                            label = label.lower()
                        this_ann = {
                            "ann": {'text': label}
                        }

                        if len(label) > self.batch_max_length:
                            continue

                        self.img_infos.append(this_ann)
                        self.filtered_index_list.append(index)
                        if len(self.filtered_index_list) == int(self.num_samples * self.used_ratio):
                            break
                    self.num_samples = len(self.filtered_index_list)

                else:
                    # rectify LMDB_Davar data
                    self.key_list = [key.decode("utf8")
                                     for key, _ in txn.cursor()
                                     if not key.decode("utf8").endswith(".IMG")]
                    # get the total number of the dataset
                    self.num_samples = len(self.key_list)

                    # load the label information, filter label according to the maxlength
                    for index, key in enumerate(self.key_list):
                        value = json.loads(txn.get(key.encode()).decode("utf8"))
                        label = value["content_ann"]["texts"][0]
                        if not self.pipeline_dict[0]["sensitive"]:
                            label = label.lower()
                        this_ann = {
                            "filename": osp.join(self.img_prefix, key),
                            "ann": {
                                'text': value["content_ann"]['texts'][0],
                                'bbox': value["content_ann"]['bboxes'][0],
                                'label': value["content_ann"]['labels'][0]
                                if 'labels' in value["content_ann"] else -1, }
                            }

                        if len(label) > self.batch_max_length:
                            continue

                        self.img_infos.append(this_ann)
                        self.filtered_index_list.append(index)
                        if len(self.filtered_index_list) == int(self.num_samples * self.used_ratio):
                            break
                    self.num_samples = len(self.filtered_index_list)

    def json_file_load(self, root):
        """

        Args:
            root (str): root path of the File、Tight、Loose dataset

        Returns:

        """

        # load the label information
        self.img_infos = self._load_annotations(root)

        # load the loose data type
        if self.data_type == 'Loose':
            # load the 'box', 'text', 'label' in label information
            self.img_infos = self._loose_to_list(self.img_infos)
        else:
            if self.phase == 'Test' and self.test_filter is not None:
                # load the 'box', 'text', 'label' in label information
                self.img_infos = self._filter_file_to_list(self.img_infos)
            else:
                # load the File data type
                if self.data_type == "File":
                    # load the 'box', 'text', 'label' in label information
                    self.img_infos = self._file_to_list(self.img_infos)
                # load the Tight data type
                elif self.data_type == "Tight":
                    # load the 'box', 'text', 'label' in label information
                    self.img_infos = self._tight_to_list(self.img_infos)
                else:
                    raise TypeError("Invalid data types !!!")

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

    def __len__(self):
        """

        Returns:
            int: sample numbers

        """
        return self.num_samples

    def get_ann_info(self, idx):
        """
        Args:
            idx (index): sampler index

        Returns:
            int: sampler index corresponding label information

        """

        return self.img_infos[idx]

    def prepare_train_img(self, idx):
        """

        Args:
            idx (int): training sample index

        Returns:
            dict: training sample after pre-process pipeline operation

        """
        if self.data_type == "LMDB_Standard":
            results = dict(index=idx, env=self.env)
        elif self.data_type == "LMDB_Davar":
            results = dict(key=self.key_list[idx], env=self.env)
        elif self.data_type == "File" or self.data_type == "Tight" or self.data_type == "Loose":
            results = dict(img_info=self.img_infos[idx], img_prefix=self.img_prefix)
        else:
            raise TypeError("Invalid data type !!!")

        results = self.pipeline(results)
        return results

    def __getitem__(self, index):
        """

        Args:
            index (int): training sample index

        Returns:
            dict: training sample after shuffle

        """
        assert index <= len(self), 'index range error'
        idx = self.filtered_index_list[index]
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                ori_idx = idx
                idx = random.randint(0, len(self)-1)
                idx = self.filtered_index_list[idx]
                if self.phase == "Train":
                    continue
                else:
                    self.img_infos[ori_idx] = self.img_infos[idx]
            return data

    def _load_annotations(self, root):
        """

        Args:
            root (str): root path of the label information

        Returns:
             File: load the json file with the label information

        """
        return mmcv.load(root)

    def _file_to_list(self, img_infos):
        """
        Args:
            img_infos (dict): the dict with the label information

        Returns:
            dict: select the training sample list

        """
        reslist = list()

        # Determine whether 'bboxes' in img_infos
        flag = 'bboxes' in str(img_infos)
        for k, value in img_infos.items():
            texts = value['content_ann']['texts']
            labels = value['content_ann'].get('labels', [None]*len(texts))
            height = img_infos[k]["height"]
            width = img_infos[k]["width"]
            if flag:
                # load the 'bboxes' information
                bboxes = value['content_ann']['bboxes']
            else:
                # 'bboxes' is not in img_infos, use the image size as 'bboxes' information
                bboxes = img_infos[k]["content_ann"].get("bboxes", [0, 0, width - 1, 0, width - 1,
                                                                    height - 1, 0, height - 1])
            cares = value['content_ann'].get('cares', [1]*len(texts))
            for i, text in enumerate(texts):
                if not cares[i] or len(bboxes[i]) != 8:   # Filter out defective boxes and polygonal boxes
                    continue
                reslist.append({
                    'filename': k,
                    'ann': {
                        'height': height,
                        'width': width,
                        'text': text if self.pipeline_dict[0]["sensitive"] else text.lower(),
                        'bbox': bboxes[i],
                        'label': labels[i]
                    }
                                })
        return reslist

    def _tight_to_list(self, img_infos):
        """
        Args:
            img_infos (dict): the dict with the label information

        Returns:
            list(dict): select the training sample list

        """
        reslist = loose_tight_to_list(img_infos, self.pipeline_dict)

        return reslist

    def _filter_file_to_list(self, img_infos):
        """
        Args:
            img_infos (dict): the dict with the label information

        Returns:
            list(dict): select the training sample list

        """
        reslist = list()

        # parameter initialization
        cares_filter = getattr(self.test_filter, 'cares', {1})
        labels_filter = getattr(self.test_filter, 'labels', None)
        minh_filter = getattr(self.test_filter, 'minh', 0)
        text_no_filter = getattr(self.test_filter, 'text_no', None)
        text_len_filter = getattr(self.test_filter, 'text_len', None)

        # Determine whether 'bboxes' in img_infos
        flag = 'bboxes' in str(img_infos)
        for k, value in img_infos.items():
            texts = value['content_ann']['texts']
            cares = value['content_ann'].get('cares', [1]*len(texts))  # ['cares']
            height = img_infos[k]["height"]
            width = img_infos[k]["width"]
            if flag:
                # load the 'bboxes' information
                bboxes = value['content_ann']['bboxes']
            else:
                # 'bboxes' is not in img_infos, use the image size as 'bboxes' information
                bboxes = img_infos[k]["content_ann"].get("bboxes", [0, 0, width - 1, 0, width - 1, 0, height - 1])

            labels = value['content_ann'].get('labels', [None]*len(texts))

            # filter the label information
            for i, text in enumerate(texts):
                if len(bboxes[i]) != 8:  # Filter the polygonal boxes
                    continue
                if minh_filter > 0:  # Filter the box whose height is smaller than 0
                    if (abs(bboxes[i][7]-bboxes[i][1]) + abs(bboxes[i][5]-bboxes[i][3])) // 2 < minh_filter:
                        continue
                if cares[i] not in cares_filter:  # filter the "uncare" data
                    continue
                if text_no_filter is not None:
                    assert isinstance(text_no_filter, (str, list))
                    if isinstance(text_no_filter, str):
                        if text_no_filter in text:
                            continue
                    elif isinstance(text_no_filter, list):
                        continue_flag = False
                        for text_no in text_no_filter:
                            if text_no in text:
                                continue_flag = True
                                break
                        if continue_flag:
                            continue
                    else:
                        raise ValueError('text_no_filter type error')
                # filter the data by the length
                if text_len_filter is not None:
                    if isinstance(text_len_filter, int):
                        if len(text) != text_len_filter:
                            continue
                    elif isinstance(text_len_filter, list):
                        assert len(text_len_filter) >= 2, text_len_filter
                        if len(text_len_filter) == 2:
                            if len(text) < text_len_filter[0] or len(text) > text_len_filter[1]:
                                continue
                        else:
                            if len(text) not in text_len_filter:
                                continue
                    elif isinstance(text_len_filter, set):
                        if len(text) not in text_len_filter:
                            continue
                    else:
                        raise TypeError('text_len_filter type error')
                if labels_filter is not None:  # filter the illegal data
                    assert isinstance(labels_filter, dict)
                    continue_flag = False
                    try:
                        for l_k, l_v in labels_filter.items():
                            if isinstance(l_v, int):
                                if labels[i][int(l_k)] != l_v:
                                    continue_flag = True
                                    break
                            elif isinstance(l_v, set):
                                if labels[i][int(l_k)] not in l_v:
                                    continue_flag = True
                                    break
                            else:
                                raise ValueError('wrong labels_filter type')
                        if continue_flag:
                            continue
                    except:
                        print(text, cares[i], labels[i])

                reslist.append({
                    'filename': k,
                    'ann': {
                        'height': height,
                        'width': width,
                        'text': text if self.pipeline_dict[0]["sensitive"] else text.lower(),
                        'bbox': bboxes[i],
                        'label': labels[i]
                    }
                })

        return reslist

    def _loose_to_list(self, img_infos):
        """
        Args:
            img_infos (dict): the dict with the label information

        Returns:
            list(dict): select the training sample list

        """

        reslist = loose_tight_to_list(img_infos, self.pipeline_dict)

        return reslist

    def evaluate(self,
                 results,
                 metric='accuracy',
                 logger=None,
                 **eval_kwargs):
        """
        Args:
            results (list): model inference result
            metric (str): model performance metric
            logger (logger): training logger
            **eval_kwargs (None): backup parameter

        Returns:
            dict: model evaluation result
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['accuracy', 'NED']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        n_correct = 0
        norm_ed = 0
        cnt_correct = 0
        cnt_flag = False

        assert len(results) == len(self), 'model prediction != length of dataset : {} != {}'.\
            format(len(results), len(self))
        length_of_data = len(self)

        labels, filenames = list(), list()

        labels.extend(copy.deepcopy([self.get_ann_info(i)['ann']['text'] for i in range(len(self))]))
        filenames.extend(copy.deepcopy([self.get_ann_info(i)['filename'] for i in range(len(self))
                                        if 'filename' in self.get_ann_info(i)]))

        # general recognition model
        if isinstance(results[0], str):
            print("\n")
            for i in range(min(length_of_data, 5)):
                print('gt: %-30s\t pred: %-30s' % (labels[i], results[i]))

        # rf_learning visual counting
        if isinstance(results[0], int):
            print("\n")
            for i in range(min(length_of_data, 5)):
                print('gt: %-30s\t length of gt:%-30s\t pred: %-30s' % (labels[i], len(labels[i]), results[i]))

        # rf_learning total
        if isinstance(results[0], tuple) and isinstance(results[0][0], str) and isinstance(results[0][1], int):
            print("\n")
            for i in range(min(length_of_data, 5)):
                print('gt: %-30s\t pred str: %-30s\t length of gt:%-30s\t  pred length:%-30s' %
                      (labels[i], results[i][0], len(labels[i]), results[i][1]))

        for pred, label in zip(results, labels):
            if isinstance(pred, str):

                # prediction filter
                pred = filter_punctuation(pred, r':(\'-,%>.[?)"=_*];&+$@/|!<#`{~\}^')
                label = filter_punctuation(label, r':(\'-,%>.[?)"=_*];&+$@/|!<#`{~\}^')

                # calculate the accuracy
                if pred == label:
                    n_correct += 1

                if len(label):
                    norm_ed += edit_distance(pred, label) / len(label)
                elif not len(label) and len(pred):
                    norm_ed += edit_distance(pred, label) / len(pred)
                else:
                    norm_ed += 0

            if isinstance(pred, int):
                # calculate the counting accuracy
                if pred == len(label):
                    n_correct += 1
                norm_ed = 0

            if isinstance(pred, tuple):
                cnt_flag = True
                if pred[1] == len(label):
                    cnt_correct += 1
                if pred[0] == label:
                    n_correct += 1

                    if len(label):
                        norm_ed += edit_distance(pred[0], label) / len(label)
                    elif not len(label) and len(pred[0]):
                        norm_ed += edit_distance(pred[0], label) / len(pred[0])
                    else:
                        norm_ed += 0

        accuracy = n_correct / float(length_of_data) * 100

        validation_dict = dict()

        if cnt_flag:
            cnt_accuracy = cnt_correct / float(length_of_data) * 100
            validation_dict["cnt_accuracy"] = cnt_accuracy

        validation_dict["accuracy"] = accuracy
        validation_dict["NED"] = norm_ed
        return validation_dict


def loose_tight_to_list(img_infos, pipeline_dict):
    """
    Args:
        img_infos (dict): dict saving the label information
        pipeline_dict (dict): pipeline information

    Returns:
        list(dict): select the training sample list
    """

    reslist = list()
    for key, value in img_infos.items():
        try:
            # load the 'bboxes' information
            texts = value['content_ann']['texts']
            if not isinstance(texts, (list, tuple)):
                texts = [texts]
            if not texts[0]:
                continue
            if 'bboxes' in img_infos[key]["content_ann"] and len(img_infos[key]["content_ann"]['bboxes'][0]) == 8:
                # load the 'bboxes' information if length of the 'bboxes' equals 8
                bboxes = img_infos[key]["content_ann"]['bboxes'][0]
            else:
                # 'bboxes' is not in img_infos, use the image size as 'bboxes' information
                height = img_infos[key]["height"]
                width = img_infos[key]["width"]
                bboxes = [0, 0, width - 1, 0, width - 1, height - 1, 0, height - 1]
        except KeyError as _:
            print(value)
            continue

        for _, text in enumerate(texts):
            reslist.append({
                'filename': key,
                'ann': {
                    'text': text if pipeline_dict[0]["sensitive"] else text.lower(),
                    'bbox': bboxes,
                    'label': value["content_ann"]['labels'][0]
                    if 'labels' in value["content_ann"] else -1,
                }
            })
    return reslist
