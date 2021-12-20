"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_loading_json.py
# Abstract       :    Implementations of davar json-type pipelines

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import re
import os.path as osp

import cv2
import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES

from .utils.loading_utils import wordmap_loader, shake_crop, shake_point, \
    scale_box, scale_box_hori_vert, get_perspective_img, crop_and_transform, rotate_and_crop, scale_point_hori_vert


def rcg_json_dataload(load_type,
                      results,
                      test_mode=False,
                      color_types=["bgr", "rgb", "gray"],
                      crop_pixel_shake=None,
                      use_lib_crop=False,
                      need_crop=False,
                      crop_only=False,
                      table=None,
                      sensitive=False,
                      fil_ops=False,
                      support_chars=False,
                      character=None,
                      abandon_unsupport=False,
                      expand_ratio=None,
                      crop_config=None):
    """
        File|Tight|Loose dataset data loading
    Args:
        load_type (str): type of data loading, including ["File", "Tight", "Loose"]
        results (dict): dict for saving the image data and labels
        test_mode (bool): whether to be train mode or test mode, Default(False)
        color_types (list(str)): color type of the images, including ["rgb", "bgr", "gray"]
        crop_pixel_shake (dict|list): coordinate pixel shape during image crop
        use_lib_crop (bool): crop images with the perspective transformation
        need_crop (bool): whether to crop the images, only supported in Tight Mode
        crop_only (bool): only to crop images without any other operation
        table (dict): translate table, produced by the function "maketrans()"
        sensitive (bool): upper or lower, default False(lower)
        fil_ops (bool): whether to filter the symbol out of the character dictionary
        support_chars (str): supported recognition character
        character (str): recognition dictionary
        abandon_unsupport (bool): whether to drop the unsupported character, only supported in File|Tight data type
        expand_ratio (float): ratios of the fixed expand
        crop_config (dict): setting of rotating images to horizontal, then cropping image patchers

    Returns:
        dict: dict for saving the processed image data and labels

    """

    # supported dataset type, including['File', 'Tight', 'Loose']
    assert load_type in ['File', 'Tight', 'Loose'], 'data_type should be File / Tight / Loose, but found ' \
                                                    + load_type

    # supported color type, including['rgb', 'bgr', 'gray']
    color_type = color_types[0]
    assert color_type in ["rgb", "bgr", "gray"], 'color_type should be rgb, bgr, gray , but found ' + color_type

    bbox = None

    if test_mode:
        phase = "Test"
    else:
        phase = "Train"

    # image path
    filename = osp.join(results['img_prefix'], results['img_info']['filename'])

    # dataset type is 'File' or 'Loose'
    if load_type != "Tight":
        bbox = results['img_info']['ann']['bbox'].copy()

    # load the label information
    text = results['img_info']['ann']['text']
    if 'label' in results['img_info']['ann']:
        results['gt_label'] = results['img_info']['ann']['label']

    # read the image data
    img = mmcv.imread(filename,
                      cv2.IMREAD_IGNORE_ORIENTATION +
                      cv2.IMREAD_COLOR)

    # read image with the different format
    if color_type == "rgb":
        # "rgb" format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_type == "gray":
        # "gray" format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color_type == "bgr":
        # "bgr" format
        pass
    else:
        Exception("Unsupported the color type !!!")

    if not isinstance(img, np.ndarray):
        print('Read Error at Path:', filename)
        return None

    if load_type == "File":
        v12 = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
        v34 = [bbox[6] - bbox[4], bbox[7] - bbox[5]]

        # transfer the bounding box order[1243->1234]
        if v12[0] * v34[0] + v12[1] * v34[1] > 0:
            bbox = bbox[:4] + bbox[6:] + bbox[4:6]
        if phase == 'Train' and crop_pixel_shake is not None:  # random expand
            bbox = shake_point(img, bbox, crop_pixel_shake)
        if phase == 'Test' and isinstance(expand_ratio, float) and expand_ratio > 1:  # fixed expand
            bbox = scale_box(bbox, img.shape[0], img.shape[1], expand_ratio)
        elif phase == 'Test' and isinstance(expand_ratio, list) and len(expand_ratio) == 2:
            bbox = scale_box_hori_vert(bbox, img.shape[0], img.shape[1], expand_ratio)
        elif phase == 'Test' and isinstance(expand_ratio, list) and len(expand_ratio) == 3:
            bbox = scale_point_hori_vert(bbox, img.shape[0], img.shape[1], expand_ratio)

        if use_lib_crop:
            img = get_perspective_img(img, bbox)  # crop image with the perspective transformation
        else:
            img = crop_and_transform(img, bbox, crop_only)
        if img.shape[0] == 0 or img.shape[1] == 0:  # filter the bounding box height or width with the 0 pixel
            return None
    elif load_type == "Tight":
        # whether to crop image or pixel shake
        if need_crop:
            bbox = results['img_info']['ann']['bbox']
            img = shake_crop(img, bbox, crop_pixel_shake, need_crop)
    else:
        # rotate and crop image
        img = rotate_and_crop(img, bbox, **crop_config)

        if not isinstance(img, np.ndarray):
            print('Read Error at Path:', filename)
            return None
        if crop_pixel_shake is not None:
            img = shake_crop(img, bbox, **crop_pixel_shake)

    if table is not None:
        label = text.translate(table)
    else:
        label = text

    # use the upper or lower character
    if not sensitive:
        label = label.lower()

    if phase == 'Train':
        if fil_ops:
            # Discard samples that contain unsupported characters
            # (transfer full-width characters to half-width character)
            if abandon_unsupport:
                for char in label:
                    if char not in support_chars:
                        return None
            else:
                # only filter unsupported character
                out_of_char = '[^' + character + ']'
                label = re.sub(out_of_char, '', label)

        if label == '':
            print('Tight Empty Label:', filename)
            return None

    results['img'] = img
    results['gt_text'] = label

    if load_type == "Loose":
        results['filename'] = filename

    return results


@PIPELINES.register_module()
class RCGLoadImageFromFile:
    """ text recognition load File type data"""
    def __init__(self,
                 character,
                 color_types=["bgr"],
                 test_mode=False,
                 sensitive=True,
                 fil_ops=False,
                 abandon_unsupport=False,
                 crop_aug=None,
                 use_lib_crop=True):

        """
            File type data loading
        Args:
            character (str): recognition dictionary
            color_types (list): color type of the images, including ["rgb", "bgr", "gray"]
            test_mode (bool): whether to be train mode or test mode, Default(False)
            sensitive (bool): upper or lower, default False(lower)
            fil_ops (bool): whether to filter the symbol out of the character dictionary
            abandon_unsupport (bool): whether to drop the unsupported character, only supported in File|Tight data type
            crop_aug (dict): setting of rotating images to horizontal, then cropping image patchers
            use_lib_crop (bool): crop images with the perspective transformation
        """

        self.color_types = color_types
        self.test_mode = test_mode
        self.sensitive = sensitive
        self.fil_ops = fil_ops
        self.abandon_unsupport = abandon_unsupport
        self.load_type = "File"

        self.use_lib_crop = use_lib_crop
        self.crop_pixel_shake = None
        self.expand_ratio = 1
        self.crop_only = False

        # parameter of random crop and fix expand
        if crop_aug is not None:
            self.use_lib_crop = crop_aug.get('use_lib_crop', False)
            self.crop_only = crop_aug.get('crop_only', False)
            self.crop_pixel_shake = crop_aug.get('shake_pixel', None)
            self.expand_ratio = crop_aug.get('expand_ratio', 1.1)

        self.character = None
        self.support_chars = None
        self.table = None

        # load the character dictionary
        self.character, self.support_chars, self.table = wordmap_loader(character, self.load_type)

    def __call__(self, results):
        """
        Args:
            results (dict): dict for saving the image data and labels

        Returns:
            dict: dict for saving the processed image data and labels
        """
        results = rcg_json_dataload(load_type=self.load_type,
                                    results=results,
                                    color_types=self.color_types,
                                    test_mode=self.test_mode,
                                    crop_pixel_shake=self.crop_pixel_shake,
                                    use_lib_crop=self.use_lib_crop,
                                    need_crop=False,
                                    crop_only=self.crop_only,
                                    table=self.table,
                                    sensitive=self.sensitive,
                                    fil_ops=self.fil_ops,
                                    support_chars=self.support_chars,
                                    character=self.character,
                                    abandon_unsupport=self.abandon_unsupport,
                                    expand_ratio=self.expand_ratio)

        return results


@PIPELINES.register_module()
class RCGLoadImageFromTight:
    """ text recognition load Tight type data"""
    def __init__(self,
                 character,
                 color_types=["bgr"],
                 test_mode=False,
                 sensitive=True,
                 fil_ops=False,
                 abandon_unsupport=False,
                 crop_aug=None,
                 need_crop=True):
        """
            Tight type data loading
        Args:
            character (str): recognition dictionary
            color_types (list): color type of the images, including ["rgb", "bgr", "gray"]
            test_mode (bool): whether to be train mode or test mode, Default(False)
            sensitive (bool): upper or lower, default False(lower)
            fil_ops (bool): whether to filter the symbol out of the character dictionary
            abandon_unsupport (bool): whether to drop the unsupported character, only supported in File|Tight data type
            crop_aug (dict): setting of rotating images to horizontal, then cropping image patchers
            need_crop (bool): whether to crop the images, only supported in Tight Mode
        """

        self.color_types = color_types
        self.test_mode = test_mode
        self.sensitive = sensitive
        self.fil_ops = fil_ops
        self.abandon_unsupport = abandon_unsupport
        self.need_crop = need_crop
        self.load_type = "Tight"

        # parameter of random crop and fix expand
        self.crop_pixel_shake = None
        if crop_aug is not None:
            self.crop_pixel_shake = crop_aug['shake_pixel']

        self.character = None
        self.support_chars = None
        self.table = None

        # load the character dictionary
        self.character, self.support_chars, self.table = wordmap_loader(character, self.load_type)

    def __call__(self, results):
        """
        Args:
            results (dict): dict for saving the image data and labels

        Returns:
            dict: dict for saving the processed image data and labels
        """
        results = rcg_json_dataload(load_type=self.load_type,
                                    results=results,
                                    color_types=self.color_types,
                                    test_mode=self.test_mode,
                                    crop_pixel_shake=self.crop_pixel_shake,
                                    use_lib_crop=False,
                                    need_crop=self.need_crop,
                                    table=self.table,
                                    sensitive=self.sensitive,
                                    fil_ops=self.fil_ops,
                                    support_chars=self.support_chars,
                                    character=self.character,
                                    abandon_unsupport=self.abandon_unsupport)

        return results


@PIPELINES.register_module()
class RCGLoadImageFromLoose:
    """ text recognition load Loose type data"""
    def __init__(self,
                 character,
                 color_types=["bgr"],
                 test_mode=False,
                 sensitive=False,
                 fil_ops=False,
                 abandon_unsupport=False,
                 crop_config={'crop_method': "crop_and_transform"},
                 crop_aug=None):
        """
            Loose type data loading
        Args:
            character (str): recognition dictionary
            color_types (list): color type of the images, including ["rgb", "bgr", "gray"]
            test_mode (bool): whether to be train mode or test mode, Default(False)
            sensitive (bool): upper or lower, default False(lower)
            fil_ops (bool): whether to filter the symbol out of the character dictionary
            abandon_unsupport (bool): whether to drop the unsupported character, only supported in File|Tight data type
            crop_config (dict): setting of rotating images to horizontal, then cropping image patchers
            crop_aug (dict): setting of rotating images to horizontal, then cropping image patchers
        """

        self.color_types = color_types
        self.test_mode = test_mode
        self.sensitive = sensitive
        self.fil_ops = fil_ops
        self.abandon_unsupport = abandon_unsupport
        self.load_type = "Loose"

        # parameter of random crop and fix expand
        self.crop_pixel_shake = None
        if crop_aug is not None:
            self.crop_pixel_shake = crop_aug['shake_pixel']

        self.character = None
        self.support_chars = None
        self.table = None

        # load the character dictionary
        self.character, self.support_chars, self.table = wordmap_loader(character, self.load_type)

        assert 'crop_method' in crop_config and crop_config['crop_method'] in \
               ("crop_and_transform", "rotate_and_crop", "perspective_crop")

        self.crop_config = crop_config

    def __call__(self, results):
        """
        Args:
            results (dict): dict for saving the image data and labels

        Returns:
            dict: dict for saving the processed image data and labels
        """
        results = rcg_json_dataload(load_type=self.load_type,
                                    results=results,
                                    color_types=self.color_types,
                                    test_mode=self.test_mode,
                                    crop_pixel_shake=self.crop_pixel_shake,
                                    use_lib_crop=False,
                                    need_crop=False,
                                    table=self.table,
                                    sensitive=self.sensitive,
                                    fil_ops=self.fil_ops,
                                    support_chars=self.support_chars,
                                    character=self.character,
                                    abandon_unsupport=self.abandon_unsupport,
                                    crop_config=self.crop_config)

        return results
