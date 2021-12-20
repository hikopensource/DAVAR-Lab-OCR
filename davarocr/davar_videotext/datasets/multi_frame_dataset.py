"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    yoro_det_dataset.py
# Abstract       :    Implementation of yoro detection dataset.

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
import os
import os.path as osp
import random

import numpy as np
import mmcv

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from davarocr.davar_common.datasets.davar_custom import DavarCustomDataset


@DATASETS.register_module()
class MultiFrameDataset(DavarCustomDataset):
    """ Implementation of the common video text dataset, which supports tasks of video text detection

       train_datalist.json:                                                        # file name
        {
            "###": "Comment",                                                      # The meta comment
            "Images/train/1.jpg": {                                             # Relative path of images
                "height": 534,                                                     # Image height
                "width": 616,                                                      # Image width
                "video": "Video_19_5_1",                                           # Image belongs to the video
                "frameID": "1",                                                    # Frame number
                "content_ann": {                                                   # Following lists have same lengths.
                    "bboxes": [[161, 48, 563, 195, 552, 225, 150, 79],             # Bounding boxes in shape of [2 * N]
                                [177, 178, 247, 203, 240, 224, 169, 198],          # where N >= 2. N=2 means the
                                [263, 189, 477, 267, 467, 296, 252, 218]]          # axis-alignedrect bounding box,
                    "cbboxes": [ [[...],[...]], [[...],[...],[...]],               # Character-wised bounding boxes
                    "cares": [1, 1, 0],                                            # If the bboxes will be cared
                    "labels": [['title'], ['code'], ['num']],                      # Labels for classification/detection
                                                                                   # task, can be int or string.
                    "texts": ['apple', 'banana', '###'],                           # Transcriptions for text recognition
                    "trackID": ["Video_19_5_1-1001", "Video_19_5_1-1002", "Video_19_5_1-1003"] # The track seq id for
                                                                                                each text instance
                    "qualities": ["HIGH", "MODERATE", "LOW"]                       # The quality of each text
                }
            },
            ....
        }
    """
    def __init__(self,
                 ann_file,
                 pipeline,
                 flow_path=None,
                 window_size=3,
                 data_root=None,
                 img_prefix='',
                 test_mode=False,
                 filter_empty_gt=True,
                 classes_config=None,
                 ):
        """
            Args:
                ann_file(str): the path to datalist.
                pipeline(list(dict)): the data-flow handling pipeline
                flow_path(str): the path to flow dir
                window_size(int): the nums of consecutive frames in a batch
                data_root(str): the root path of the dataset
                img_prefix(str): the image prefixes
                test_mode(boolean): whether in test mode
                filter_empty_gt(boolean): whether to filter out image without ground-truthes.
                classes_config(str): the path to classes config file, used to transfer 'str' labels into 'int'
        """

        self.indices = []
        super().__init__(ann_file, pipeline, data_root=data_root, img_prefix=img_prefix, test_mode=test_mode,
                         filter_empty_gt=filter_empty_gt)
        self.ann_file = ann_file
        self.data_root = data_root
        self.flow_path = flow_path
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.window_size = window_size
        self.indices = []
        self.data_infos = []
        random.seed(1)

        # Join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_files = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        # Load annotations
        data_infos = self.load_annotations(self.ann_file)
        self.data_infos = self._cvt_list(data_infos)

        # Filter images with no annotation during training and generate indices for sampler
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            self.data_infos = self.group_sort_process(self.data_infos)
            self.indices = self.prepare_index(self.data_infos)

        print('len(self.data_infos)', len(self.data_infos))

        # Processing pipeline
        self.pipeline = Compose(pipeline)
        if classes_config is not None:
            self.classes_config = mmcv.load(classes_config)
        else:
            self.classes_config = None

        self.flag = np.zeros(len(self), dtype=np.uint8)

        # Set group flag for the sampler, in video text task, all images are same flag
        for i in range(len(self)):
            self.flag[i] = 1

    def _filter_imgs(self, min_size=32):
        """Filter images too small and images without annoations

        Args:
            min_size(int): minimum supported image size

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

    def __len__(self):

        return len(self.indices)

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

        # Update key info
        for key in img_info.keys():
            tmp_dict = dict()
            tmp_dict["filename"] = key
            tmp_dict["height"] = img_info[key]["height"]
            tmp_dict["width"] = img_info[key]["width"]
            tmp_dict["ann"] = img_info[key]["content_ann"]
            tmp_dict["video"] = img_info[key].get("video", None)
            tmp_dict["frameID"] = img_info[key].get("frameID", None)
            result_dict.append(tmp_dict)

        return result_dict

    def pre_pipeline(self, results):
        """ Prepare pipelines. Integrated with some preset keys, like `bbox_fields`, `cbbox_fields`

        Args:
            results(list(dict) | dict): original data flow

        Returns:
            dict: updated data flow
        """

        def integrate(result):
            """ Common integration. Integrated with some preset keys, like `bbox_fields`, `cbbox_fields`  """
            result['cbbox_fields'] = []
            result['img_prefix'] = self.img_prefix
            result['bbox_fields'] = []
            result['mask_fields'] = []

            return result

        # Deal with results(list(dict)) contains multiple instances
        if isinstance(results, list):
            for data in results:
                data = integrate(data)
                if data['img_info'].get("video", None):
                    data["video"] = data['img_info']["video"]
                if data['img_info'].get("frameID", None):
                    data["frameID"] = data['img_info']["frameID"]
                if "flow" in data['img_info']:
                    data["flow"] = data['img_info']["flow"]

        # Deal with results(dict) contains single instances
        elif isinstance(results, dict):
            results = integrate(results)
        else:
            raise NotImplementedError

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \ introduced by pipeline.
        """
        choice_indices = self.indices[idx]

        results = []
        for index in choice_indices:

            # Loading each instance info
            img_info = self.data_infos[index]
            ann_info = self.get_ann_info(index)

            # Loading optical flow
            flow_info = self.load_flows(img_info)
            img_info['flow'] = flow_info

            result = dict(img_info=img_info, ann_info=ann_info, classes_config=self.classes_config)
            results.append(result)

        # Pipeline
        self.pre_pipeline(results)
        results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by pipeline.

        """
        # Loading each instance info
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        # Loading optical flow
        flow_info = self.load_flows(img_info)
        img_info['flow'] = flow_info

        results = dict(img_info=img_info, ann_info=ann_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]

        # Pipeline
        self.pre_pipeline(results)
        return self.pipeline(results)

    def check_flows_file(self, data_infos):
        """Check flows file whether exists.

        Args:
            data_infos (list(dict)): data.

        """
        for img_info in data_infos:
            video = img_info['video']
            frame_id = img_info['frameID']
            assert osp.isfile(osp.join(self.flow_path, video, str(frame_id) + '.npz'))

    def load_flows(self, img_info):
        """Get flow data.

         Args:
             img_info (dict): img info for target img.

         Returns:
             numpy array: flow data.e.g:[window size - 1 x H x W x 2]

         """
        video = img_info['video']
        frame_id = img_info['frameID']
        data = np.load(os.path.join(self.flow_path, video, str(frame_id) + '.npz'))
        flow = data['arr_0']
        return flow

    def group_sort_process(self, data_infos):
        """Sort the data info according to the video name and frame id.

         Args:
             data_infos (list(dict)): all data.

         Returns:
             list(dict): sorted data.
         """
        # Save the sorted data
        video_group = {}

        for data in data_infos:

            # Read each video's frames
            video_name = data['video']
            if not video_name in video_group.keys():
                video_group[video_name] = [data]
            else:
                video_group[video_name].append(data)

        sorted_img_infos = []

        # Sort the data by frame number
        for value in video_group.values():
            sorted_value = sorted(value, key=lambda x: int(x['frameID']))
            sorted_img_infos += sorted_value

        data_infos = sorted_img_infos
        return data_infos

    def prepare_index(self, data_infos):
        """Generate the indices to fetch data, make sure the data in batch belong to same video and is consecutive
        frames.

         Args:
             data_infos (list(dict)): data info.

         Returns:
             list(list()): indices, like [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]].
         """
        indices = []

        # Generating the data list indices which the list in list indicates window size consecutive frames' indices
        for i, img_info in enumerate(data_infos):
            video = img_info['video']
            index = [i]
            # Reach the end of data
            if i >= (len(data_infos) - self.window_size + 1):
                continue

            # Fetch the consecutive frame index
            for j in range(1, self.window_size):
                next_video = data_infos[i + j]['video']
                if next_video == video:
                    index.append(i + j)
                else:
                    break

            if len(index) == self.window_size:
                indices.append(index)

        return indices
