"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ner_transforms.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
from mmdet.datasets.builder import PIPELINES
from davarocr.davar_common.core import build_converter


@PIPELINES.register_module()
class NERTransform:
    """Convert text to ID and entity in ground truth to label ID. The masks and
    tokens are generated at the same time.
    """
    def __init__(self, label_converter, with_label=True, keys=[]):
        """
        Args:
            label_converter (dict): Convert text to ID and entity
            in ground truth to label ID.
            with_label (bool): whether convert entities label to ids.
            keys(list): maintain keys for results.
        """
        self.label_converter = build_converter(label_converter)
        self.with_label = with_label
        self.keys = keys

    def __call__(self, results):
        res = self.label_converter.convert_text2id(results)
        if self.with_label:
            labels = results['token_labels']
            labels_res = self.label_converter.convert_entity2label(labels)
            res.update(labels_res)
        if self.keys:
            for key in self.keys:
                if key in res:
                    raise KeyError
                res.update({key:results[key]})
        return res
