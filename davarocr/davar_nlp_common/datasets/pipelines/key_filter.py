"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    key_filter.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-02-22
##################################################################################################
"""
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class KeyFilter:
    """filter key in results.

    Args:
        filter_keys(list): filter keys for results.
    """

    def __init__(self, filter_keys=None):
        if filter_keys is None:
            self.filter_keys = []
        else:
            self.filter_keys = filter_keys

    def __call__(self, results):
        if self.filter_keys:
            for key in self.filter_keys:
                results.pop(key)
        return results




