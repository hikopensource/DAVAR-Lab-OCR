"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_loading_lmdb.py
# Abstract       :    Implementations of davar lmdb-type pipelines

# Current Version:    1.0.1
# Date           :    2022-04-27
##################################################################################################
"""
import re
import json
import os.path as osp

import six
from PIL import Image
import numpy as np
from mmdet.datasets import PIPELINES

from .utils.loading_utils import wordmap_loader, shake_crop


def rcg_lmdb_dataload(load_type,
                      results,
                      table=None,
                      test_mode=False,
                      color_types=["bgr", "rgb", "gray"],
                      crop_pixel_shake=None,
                      img_h=32,
                      img_w=100,
                      fil_ops=False,
                      character=None,
                      sensitive=False,
                      abandon_unsupport=None,
                      support_chars=None):
    """
        LMDB_Davar|LMDB_Standard dataset data loading
    Args:
        load_type (str): type of data loading, including ["LMDB_Davar", "LMDB_Standard"]
        results (dict): dict for saving the image data and labels
        table (dict): translate table, produced by the function "maketrans()"
        test_mode (bool): whether to be train mode or test mode, Default(False)
        color_types (list): color type of the images, including ["rgb", "bgr", "gray"]
        crop_pixel_shake (list): coordinate pixel shape during image crop
        img_h (int): image height
        img_w (int): image width
        fil_ops (bool): whether to filter the symbol out of the character dictionary
        character (str): recognition dictionary
        sensitive (bool): upper or lower, default False(lower)
        abandon_unsupport (bool): whether to drop the unsupported character, only supported in LMDB data type
        support_chars (str): supported recognition character

    Returns:
        dict: dict for saving the processed image data and labels
    """

    # supported dataset type, including['LMDB_Standard', 'LMDB_Davar']
    assert load_type in ['LMDB_Standard', 'LMDB_Davar'], 'data_type should be LMDB_Standard or LMDB_Davar , but found '\
                                                         + load_type

    index = None
    key = None

    if test_mode:
        phase = "Test"
    else:
        phase = "Train"

    # # supported color type, including['rgb', 'bgr', 'gray']
    color_type = color_types[0]
    assert color_type in ["rgb", "bgr", "gray"], 'color_type should be rgb, bgr, gray , but found ' + color_type

    env = results['env']
    if load_type == "LMDB_Standard":
        index = results['index']
    else:
        key = results['key']

    with env.begin(write=False) as txn:
        if load_type == "LMDB_Standard":
            # dataset type is open-source LMDB format
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
        else:
            # dataset type is LMDB_Davar format
            value = json.loads(txn.get(key.encode()).decode("utf8"))
            label = value["content_ann"]["texts"][0]
            bbox = value["content_ann"]["bboxes"][0]
            if table is not None:
                label = label.translate(table)

            # filter the illegal label, e.g("###"), length of bounding box does not equal 8, vertical text
            if phase == 'Train':
                if '###' in label:
                    return None
                if len(bbox) != 8:
                    return None
                if len(label) > 1 and abs(bbox[2] - bbox[0]) * 3 < abs(bbox[7] - bbox[1]) * 2:
                    return None
            imgbuf = txn.get((key + ".IMG").encode())

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        try:
            # read image with the different format
            if color_type == "rgb":
                # "rgb" format
                img = Image.open(buf).convert('RGB')  # for color image
            elif color_type == "gray":
                # "gray" format
                img = Image.open(buf).convert('L')
            elif color_type == "bgr":
                # "bgr" format
                img = Image.open(buf).convert('RGB')
                img = np.array(img)
                img = img[:, :, ::-1]
                img = Image.fromarray(img)
            else:
                Exception("Unsupported the color type !!!")

            img = np.array(img)
            if load_type == "LMDB_Davar":
                # crop image or pixel shake
                img = shake_crop(img, bbox, crop_pixel_shake)
        except IOError:
            if load_type == "LMDB_Standard":
                print('Corrupted image for ' + str(index))
                # make dummy image and dummy label for corrupted image.
                if color_type == "rgb":
                    img = Image.new('RGB', (img_w, img_h))
                elif color_type == "gray":
                    img = Image.new('L', (img_w, img_h))
                elif color_type == "bgr":
                    img = Image.new('RGB', (img_w, img_h))
                    img = np.array(img)
                    img = img[:, :, ::-1]
                    img = Image.fromarray(img)
                else:
                    Exception("Unsupported the color type !!!")
                label = '[dummy_label]'
                img = np.array(img)
            else:
                print('Corrupted image')

        # use the upper or lower character
        if not sensitive:
            label = label.lower()

        if phase == "Train":
            if load_type == "LMDB_Standard":
                if fil_ops:
                    # only filter unsupported character
                    out_of_char = '[^' + str(character) + ']'
                    label = re.sub(out_of_char, '', label)
            else:
                if fil_ops:
                    # Discard samples that contain unsupported characters
                    # (transfer full-width characters to half-width character)
                    if abandon_unsupport:
                        for char in label:
                            if char not in support_chars:
                                return None
                    else:
                        # Discard samples that contain unsupported characters
                        out_of_char = '[^' + character + ']'
                        label = re.sub(out_of_char, '', label)

            if label == '':
                # print('LMDB_Davar Empty Label:',key)
                return None

        results['img'] = img
        results['gt_text'] = label

    return results


@PIPELINES.register_module()
class LoadImageFromLMDB:
    """ text recognition load open-source LMDB type data"""
    def __init__(self,
                 character='0123456789abcdefghijklmnopqrstuvwxyz',
                 test_mode=False,
                 color_types=["bgr"],
                 img_h=32,
                 img_w=100,
                 sensitive=True,
                 fil_ops=True):
        """
            open-source LMDB type data loading
        Args:
            character (str): recognition dictionary
            test_mode (bool): whether to be train mode or test mode, Default(False)
            color_types (list): color type of the images, including ["rgb", "bgr", "gray"]
            img_h (int): image height
            img_w (int): image width
            sensitive (bool): upper or lower, default False(lower)
            fil_ops (bool): whether to filter the symbol out of the character dictionary
        """

        self.test_mode = test_mode
        self.color_types = color_types
        self.img_h = img_h
        self.img_w = img_w

        self.sensitive = sensitive
        self.fil_ops = fil_ops
        self.load_type = "LMDB_Standard"

        # load the character file
        if osp.exists(character):
            # character is file
            with open(character, 'r', encoding='utf8') as character_file:
                character = character_file.readline().strip()
                self.character = character
        else:
            # load the character
            self.character = character

    def __call__(self, results):
        """
        Args:
            results (dict): dict for saving the image data and labels

        Returns:
            dict: dict for saving the processed image data and labels
        """
        results = rcg_lmdb_dataload(load_type=self.load_type,
                                    results=results,
                                    test_mode=self.test_mode,
                                    color_types=self.color_types,
                                    img_h=self.img_h,
                                    img_w=self.img_w,
                                    fil_ops=self.fil_ops,
                                    character=self.character,
                                    sensitive=self.sensitive,)
        return results


@PIPELINES.register_module()
class RCGLoadImageFromLMDB:
    """ text recognition load davar LMDB type data"""
    def __init__(self,
                 character,
                 color_types=["bgr"],
                 test_mode=False,
                 sensitive=True,
                 fil_ops=False,
                 abandon_unsupport=False,
                 crop_aug=None):
        """
            LMDB_Davar type data loading
        Args:
            character (str): recognition dictionary
            color_types (list): color type of the images, including ["rgb", "bgr", "gray"]
            test_mode (bool): whether to be train mode or test mode, Default(False)
            sensitive (bool): upper or lower, default False(lower)
            fil_ops (bool): whether to filter the symbol out of the character dictionary
            abandon_unsupport (bool): whether to drop the unsupported character, only supported in LMDB_Davar data type
            crop_aug (dict): setting about coordinate pixel shape during image crop
        """

        # parameter initialization
        self.color_types = color_types
        self.test_mode = test_mode
        self.sensitive = sensitive
        self.fil_ops = fil_ops
        self.abandon_unsupport = abandon_unsupport
        self.load_type = "LMDB_Davar"

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
        results = rcg_lmdb_dataload(load_type=self.load_type,
                                    results=results,
                                    table=self.table,
                                    test_mode=self.test_mode,
                                    color_types=self.color_types,
                                    crop_pixel_shake=self.crop_pixel_shake,
                                    fil_ops=self.fil_ops,
                                    character=self.character,
                                    sensitive=self.sensitive,
                                    abandon_unsupport=self.abandon_unsupport,
                                    support_chars=self.support_chars)

        return results
