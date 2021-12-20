"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    yoro_sampler.py
# Abstract       :    The sampler used to sample data to metric learning,e.g: triplet loss

# Current Version:    1.0.0
# Date           :    2021-06-28
##################################################################################################
"""
from __future__ import division
import random

from torch.utils.data import Sampler

from davarocr.davar_common.datasets.builder import SAMPLER


@SAMPLER.register_module()
class MetricSampler(Sampler):
    """ Implementation of metric learning sampler"""

    def __init__(self, dataset, shuffle=True, samples_per_gpu=1):
        """This sampler is used for video text metric learning: anchor, positive, negative

        Args:
            dataset(dataset): dataset for sampling
            shuffle(bool): whether to shuffle data
            samples_per_gpu (int): image numbers in each gpu
        """
        super().__init__(dataset)
        random.seed(1)
        assert samples_per_gpu % 3 == 0

        self.dataset = dataset
        self.group_indices = dict()
        self.shuffle = shuffle

        for dataset_ in self.dataset.datasets:
            self.group_indices.update(self.group_index(dataset_.img_infos))

        # If one video has only one track, then choose the negative samples for different video
        for key, value in self.group_indices.items():
            if len(value.keys()) == 1:
                print(key, "this video has only one track, chose negative sample from diff video")

        # Divide the samples per gpu into 3 parts: anchor, positive, negative
        self.anchor_nums_per_gpu = samples_per_gpu // 3

        # Make sure the total size can be divided by 3
        self.left_length = len(self.dataset) - (len(self.dataset) // self.anchor_nums_per_gpu) * \
                                self.anchor_nums_per_gpu

        self.num_samples = self.anchor_nums_per_gpu - self.left_length + len(self.dataset)

    # Init when each epoch begins
    def __iter__(self):
        # Fetch the anchor indices in dataset
        anchor_indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(anchor_indices)

        # Padding the anchor to make sure it can be divided by anchor_nums_per_gpu
        anchor_indices += random.sample(anchor_indices, self.anchor_nums_per_gpu - self.left_length)
        assert len(anchor_indices) % self.anchor_nums_per_gpu == 0

        # Generating positive and negative indices
        positive_indices, negative_indices = self.generate_apn(anchor_indices)

        print("length of anchor_indices, positive_indices, negative_indices",  len(anchor_indices),
              len(positive_indices), len(negative_indices))

        # Make sure three parts length are equal
        assert len(anchor_indices) == len(positive_indices) == len(negative_indices)

        # Concat the indices by [number of anchor_nums_per_gpu anchor, number of anchor_nums_per_gpu positive,
        # number of anchor_nums_per_gpu negative, number of anchor_nums_per_gpu anchor, ...]
        indices = self.concat_apn(anchor_indices, positive_indices, negative_indices)

        assert len(indices) == len(anchor_indices) * 3

        return iter(indices)

    def __len__(self):
        return self.num_samples * 3

    def generate_apn(self, anchor_indices):
        """Generate the anchor, positive, negative indices

        Args:
            anchor_indices(list): the indices of anchor

        Returns:
            positive_indices(list): the indices of positive
            negative_indices(list): the indices of negative
        """
        positive_indices = []
        negative_indices = []

        for index in anchor_indices:
            for video_key, value in self.group_indices.items():
                for track_key in value.keys():

                    # Find which track this  anchor belongs to
                    if index in value[track_key]:

                        # this case means that the video has only one track sequence, choose negative sample from diff
                        # video
                        if len(value.keys()) == 1:

                            # Random choose video
                            negative_video_key = random.choice(list(self.group_indices.keys()))
                            while negative_video_key == video_key:
                                negative_video_key = random.choice(list(self.group_indices.keys()))

                            # Random choose negative track in video
                            negative_track_key = random.choice(list(self.group_indices[negative_video_key].keys()))
                            negative_indices.append(random.choice(
                                self.group_indices[negative_video_key][negative_track_key]))

                        # choose positive sample from same track sequence, choose negative sample from different track
                        # sequence
                        else:
                            # Random choose negative track in same video
                            negative_track_key = random.choice(list(value.keys()))
                            while negative_track_key == track_key:
                                negative_track_key = random.choice(list(value.keys()))
                            negative_indices.append(random.choice(value[negative_track_key]))

                        # Random choose positive instance in same track
                        positive_indices.append(random.choice(value[track_key]))

        return positive_indices, negative_indices

    def concat_apn(self, anchor_indices, positive_indices, negative_indices):
        """ Concat the anchor indices, positive indices, negative indices

        Args:
            anchor_indices(list): the indices of anchor
            positive_indices(list): the indices of positive
            negative_indices(list): the indices of negative

        Returns:
            list(): concat list
        """
        indices = list()
        for i in range(0, len(anchor_indices), self.anchor_nums_per_gpu):
            indices += anchor_indices[i:i + self.anchor_nums_per_gpu]
            indices += positive_indices[i:i + self.anchor_nums_per_gpu]
            indices += negative_indices[i:i + self.anchor_nums_per_gpu]
        return indices


    def group_index(self, data_infos):
        """ Record  each data index in data_infos in a dict which keys are video name and track id

        Args:
           data_infos(list): Data flow used in YORORCGDataset.

        Returns:
           dict: grouped data, each video and track id record the instances index in data_infos
        """
        track_dict = dict()

        for i, instance in enumerate(data_infos):
            # Fetch the track id for each text, the track id is like "video_name-text_id"
            track_id = instance['ann']['trackID']

            video = track_id.split('-')[0]

            # Create a new video name key
            if video not in track_dict.keys():
                track_dict[video] = dict()

            # Create a new track id key
            if track_id not in track_dict[video].keys():
                track_dict[video][track_id] = []

            # Append the index
            track_dict[video][track_id].append(i)

        return track_dict
