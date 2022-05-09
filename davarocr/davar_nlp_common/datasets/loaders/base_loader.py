"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    loader.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-08-02
##################################################################################################
"""
import os.path as osp
from ..builder import LOADERS


@LOADERS.register_module()
class BaseLoader:
    """ Load annotation from annotation file
    """
    def __init__(self, ann_file):
        """
        Args:
            ann_file (str,list,tuple): Annotation file path or path list.
        """
        assert isinstance(ann_file, (str, list, tuple))
        assert osp.exists(ann_file), f'{ann_file} is not exist'
        self.ori_data_infos = self._load(ann_file)

    def __len__(self):
        return len(self.ori_data_infos)

    def _load(self, ann_file):
        """Load annotation file."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Retrieve anno info of one instance with dict format."""
        return self.ori_data_infos[index]

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self):
            data = self[self._n]
            self._n += 1
            return data
        raise StopIteration
