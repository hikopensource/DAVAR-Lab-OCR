"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    loading.py
# Abstract       :    Definition of video text detection data formation process

# Current Version:    1.0.0
# Date           :    2021-05-31
##################################################################################################
"""
from mmdet.datasets.builder import PIPELINES
from davarocr.davar_common.datasets import DavarLoadImageFromFile, DavarLoadAnnotations


@PIPELINES.register_module()
class ConsistLoadImageFromFile(DavarLoadImageFromFile):
    """ Same with DavarLoadImageFromFile, the only difference is ConsistLoadImageFromFile support results(list) contain
    multiple instance
    """

    def __call__(self, results):
        """ Main process

        Args:
            results(dict | list(dict)): Data flow used in YORODETDataset

        Returns:
            dict | list(dict): Data flow used in YORODETDataset

        """

        # Deal with results(dict) contains single instance
        if isinstance(results, dict):
            results = super().__call__(results)
            return results

        # Deal with results(list(dict)) contains multiple instances
        results_ = []
        for instance in results:
            instance = super().__call__(instance)
            results_.append(instance)
        return results_


@PIPELINES.register_module()
class ConsistLoadAnnotations(DavarLoadAnnotations):
    """same with DavarLoadAnnotations, the only difference is ConsistLoadAnnotations support results(list)
    contain multiple instance
    """
    def __call__(self, results):
        """ Main process.

        Args:
            results(dict | list(dict)): Data flow used in YORODETDataset.

        Returns:
            dict | list(dict): output data flow.

        """
        # Deal with results(dict) contains single instance
        if isinstance(results, dict):
            results = super().__call__(results)

            return results

        # Deal with results(list(dict)) contains multiple instances
        results_ = []
        for instance in results:
            instance = super().__call__(instance)
            results_.append(instance)

        return results_
