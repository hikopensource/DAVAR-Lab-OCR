"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    sliding_window_test.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.compose import Compose


def convert_int(x):
    try:
        x = int(x)
    except:
        pass
    return x


@PIPELINES.register_module()
class SlidingWindow:
    """
    sliding window test

    Args:
        transforms(list): Transforms to apply in each test
        max_len(int): the text's max length
        truncation(bool): is truncation, if True, maintain the first only.
        stride(int): sliding stride size.
    """
    def __init__(self,
                 transforms: list,
                 max_len: int,
                 truncation: bool,
                 stride: int,
                 sign_tokens=[],
                 keys=[]):
        self.transforms = Compose(transforms)#Transforms to apply in each test
        self.max_len = max_len#the text's max length
        self.truncation = truncation#is truncation
        self.stride = stride# sliding stride setting
        self.sign_tokens = sign_tokens
        self.keys = keys
        assert isinstance(self.max_len, int)
        assert isinstance(self.truncation, bool)
        assert isinstance(self.stride, int)
        assert self.stride > 0, self.stride

    def __call__(self, results):
        """Call function to apply transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """
        aug_data = []
        lines = []
        texts = results['tokens']
        if self.truncation:
            texts = texts[:self.max_len]
            text_range = (0, self.max_len)
            temp = {"tokens":texts, "range":text_range}
            if 'id' in results:
                temp.update({'id':convert_int(results['id'])})
            if self.keys:
                for _key in self.keys:
                    key_value = results[_key]
                    temp.update({_key:key_value[text_range[0]:text_range[1]]})
            lines.append(temp)
        else:
            if not self.sign_tokens:
                for i in range(0, len(texts), self.stride):
                    if i + self.max_len > len(texts):
                        text = texts[i:len(texts)]
                        text_range = (i, len(texts))
                    else:
                        text = texts[i:i+self.max_len]
                        text_range = (i, i+self.max_len)
                    temp = {"tokens":text, "range":text_range}
                    if 'id' in results:
                        temp.update({'id':convert_int(results['id'])})
                    if self.keys:
                        for _key in self.keys:
                            key_value = results[_key]
                            temp.update({_key:key_value[text_range[0]:text_range[1]]})
                    lines.append(temp)
                
            else:
                pass

        for line_results in lines:
            _results = line_results.copy()
            data = self.transforms(_results)
            aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'max_len={self.max_len}, truncation={self.truncation}, '
        repr_str += f'stride={self.stride})'
        return repr_str
