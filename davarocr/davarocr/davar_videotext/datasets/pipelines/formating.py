"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    formating.py
# Abstract       :    Definition of video text detetcion data formating process

# Current Version:    1.0.0
# Date           :    2021-06-02
##################################################################################################
"""
from mmdet.datasets.builder import PIPELINES

from davarocr.davar_common.datasets.pipelines import DavarCollect
from davarocr.davar_det.datasets import SegFormatBundle


@PIPELINES.register_module()
class ConsistCollect(DavarCollect):
    """ Collect specific data from the data flow (results)"""
    def __init__(self,
                 keys,
                 meta_keys=('filename',  'ori_shape', 'img_shape',
                            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                            'img_norm_cfg')):

        """

        Args:
            keys(list[str]): keys that need to be collected
            meta_keys(tuple): keys of img_meta that need to be collected. e.g.,
                            - "filename": path to the image
                            - "ori_shape": original image shape
                            - "img_shape": image shape, (h, w, c).
                            - "pad_shape": image shape after padding
                            - "scale_factor": the scale factor of the re-sized image to the original image
                            - "flip": whether the image is flipped
                            - "flip_direction": the flipped direction
                            - "img_norm_cfg": configuration of normalizations
        """
        super().__init__(keys=keys, meta_keys=meta_keys)

    def __call__(self, results):
        """ Main process of ConsistCollect

        Args:
            results(dict | list(dict)): input data flow

        Returns:
            dict | list(dict): collected data informations from original data flow
        """

        # Deal with one sample when data loader call 'get_item' func which only return one sample
        if isinstance(results, dict):
            for key in self.meta_keys:
                if key in results['img_info']:
                    results[key] = results['img_info'][key]
            data = super().__call__(results)

        # Deal with multi samples when data loader call 'get_item' func which return multi samples
        elif isinstance(results, list):
            data = []
            for instance in results:
                for key in self.meta_keys:
                    if key in results['img_info']:
                        results[key] = results['img_info'][key]
                instance = super().__call__(instance)
                data.append(instance)
        else:
            raise TypeError("Nonsupport type {} of results".format(type(results)))

        return data


@PIPELINES.register_module()
class ConsistFormatBundle(SegFormatBundle):
    """ same with SegFormatBundle, the only difference is that ConsistFormatBundle can deal with multi samples
    in results
    """
    def __call__(self, results):
        """ Main process of davar_collect

        Args:
            results(dict | list(dict)): input data flow

        Returns:
            dict | list(dict): collected data informations from original data flow

        """

        # Deal with one sample when data loader call 'get_item' func which only return one sample
        if isinstance(results, dict):
            results = super().__call__(results)
            return results

        # Deal with multi samples when data loader call 'get_item' func which return multi samples
        results_ = []
        for instance in results:
            instance = super().__call__(instance)
            results_.append(instance)
        return results_
