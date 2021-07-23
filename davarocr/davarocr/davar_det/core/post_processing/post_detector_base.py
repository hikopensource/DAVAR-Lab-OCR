"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    post_detector_base.py
# Abstract       :    Base post-processing class for detectors

# Current Version:    1.0.0
# Date           :    2020-05-31
#####################################################################################################
"""
from abc import ABCMeta, abstractmethod


class BasePostDetector:
    """Base method of post-processing for detectors. Contains method of
       post_processing(batch_result):do post processing
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def post_processing(self, batch_result, **kargs):
        """ Abstract method need to be implemented"""
        pass

    def __call__(self, batch_result, **kargs):
        """ Main process of post processing"""
        return self.post_processing(batch_result, **kargs)
