"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_loading.py
# Abstract       :    Definition of common data format loading,
                      corresponding to the DavarCustomDataset

# Current Version:    1.0.1
# Date           :    2021-05-20
# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
import warnings
import os.path as osp
import mmcv
import cv2
import pycocotools.mask as maskUtils
import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.core import BitmapMasks, PolygonMasks


@PIPELINES.register_module()
class DavarLoadImageFromFile():
    """Loading image from file, add features of
       - load from nd.array
       - fix the bugs that orientation problem of cv2 reading.
    """

    def __init__(self,
                 decode_from_array=False,
                 to_float32=False):
        """ Initialization

        Args:
            decode_from_array (boolean): directly load image data from nd.array
            to_float32(boolean): transfer image data into float32
        """
        self.decode_from_array = decode_from_array
        self.to_float32 = to_float32

    def __call__(self, results):
        """ Main process

        Args:
            results(dict): Data flow used in DavarCustomDataset

        Returns:
            results(dict): Data flow used in DavarCustomDataset
        """
        if self.decode_from_array:
            data_array = results['img_info']
            assert isinstance(data_array, np.ndarray)
            data_list = [data_array[i] for i in range(data_array.size)]
            data_str = bytes(data_list)
            data_str = data_str.decode()
            data_list = data_str.split('&&')
            results['img_info'] = dict()
            results['img_info']['filename'] = data_list[0]
            results['img_info']['height'] = int(data_list[1])
            results['img_info']['width'] = int(data_list[2])

        if 'img_prefix' in results:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        elif 'img_info' in results:
            filename = results['img_info']['filename']
        else:
            filename = results['img']

        # Fix the problem of reading image reversely
        img = mmcv.imread(filename, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)

        if not isinstance(img, np.ndarray):
            print("Reading Error at {}".format(filename))
            return None

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(self.to_float32)


@PIPELINES.register_module()
class DavarLoadAnnotations():
    """The common loading function, used by DavarCustomDataset. An example is as follows

        train_datalist.json:                                                        # file name
        {
            "###": "Comment",                                                      # The meta comment
            "Images/train/img1.jpg": {                                             # Relative path of images
                "height": 534,                                                     # Image height
                "width": 616,                                                      # Image width
                "content_ann": {                                                   # Following lists have same lengths.
                    "bboxes": [[161, 48, 563, 195, 552, 225, 150, 79],             # Bounding boxes in shape of [2 * N]
                                [177, 178, 247, 203, 240, 224, 169, 198],          # where N >= 2. N=2 means the
                                [263, 189, 477, 267, 467, 296, 252, 218],          # axis-alignedrect bounding box
                                [167, 211, 239, 238, 232, 256, 160, 230],
                                [249, 227, 389, 278, 379, 305, 239, 254],
                                [209, 280, 382, 343, 366, 384, 194, 321]],
                    "cbboxes": [ [[...],[...]], [[...],[...],[...]],               # Character-wised bounding boxes
                    "cares": [1, 1, 1, 1, 1, 0],                                   # If the bboxes will be cared
                    "labels": [['title'], ['code'], ['num'], ['value'], ['other]], # Labels for classification/detection
                                                                                   # task, can be int or string.
                    "texts": ['apple', 'banana', '11', '234', '###'],              # Transcriptions for text recognition
                }
                "content_ann2":{                                                   # Second-level annotations
                    "labels": [[1],[2],[1]]
                }
                "answer_ann":{                                                   # Structure information k-v annotations
                    "keys": ["title", "code", "num","value"],
                    "value": [["apple"],["banana"],["11"],["234"]]
                }
            },
            ....
        }
    """
    def __init__(self,
                 with_bbox=False,
                 with_poly_bbox=False,
                 with_poly_mask=False,
                 with_care=False,
                 with_label=False,
                 with_multi_label=False,
                 with_text=False,
                 with_cbbox=False,
                 bieo_labels=None,
                 text_profile=None,
                 label_start_index=0,
                 poly2mask=True,
                 only_quad=False
                 ):
        """ Parameter initialization

        Args:
            with_bbox(boolean):       Whether to parse and load the bbox annotation. Used in situations that boxes are
                                      labeled as 2-point format, e.g., [[x_min, y_min, x_max, y_max],...],
                                      `results['gt_bboxes']` and `results['gt_bboxes_ignore']` will be updated.
            with_poly_bbox(boolean):  Whether to parse and load the bbox annotation. Used in situations that boxes are
                                      labeled as 2N-point format, e.g., [[x_1, y_1, x_2, y_2,..., x_n, y_n],...],
                                      `results['gt_poly_bboxes']` & `results['gt_poly_bboxes_ignore']` will be updated.
            with_poly_mask(boolean):  Whether to parse and load the mask annotation according to 'bboxes'.
                                      `results['gt_masks']` will be updated.
            with_care(boolean):       Whether to parse and load NOT_CARE label. The 'care' label decide whether a bbox
                                      is participant in training. e.g. [1, 0, 1, 1]. `results['cares']` will be updated.
            with_label(boolean):      Whether to parse and load labels (labels can be int or strings). Used in
                                      situations that labels are in 1-dimensional vectors as [[1], [3], [4], [2], ...].
                                      `results['gt_labels']"`and `results['gt_labels_ignore']` will be updated.
            with_multi_label(boolean):Whether to parse and load multiple labels (labels can be int or strings). Used in
                                      situations that labels are labeled in N-dimensional vectors like
                                      [[1,2], [3,5], [4,6], [2,2], ...]. `results['gt_labels']` and
                                      `results['gt_labels_ignore']` will be updated.
            with_text(boolean):       Whether to parse and load text transcriptions. e.g., ["apple", "banana", ...]
                                      `results['gt_texts']` and `results['gt_text']` will be updated.
            with_cbbox(boolean):      Whether to parse and load the character-wised bbox annotation.
                                      e.g., [[[x1, y1, x2, y2, x3, y3, x4, y4], [] ,[]], [[],[],[]]...],
                                      `results['gt_cbboxes']` and `results['gt_cbboxes_ignore']` will be updated.
            text_profile(dict):       Configuration of text loading, including:.
                                      text_profile=dict(
                                         text_max_length=25,   # maximum suppored text length
                                         sensitive='upper',    # whether to transfer character into "upper" or "lower" ,
                                                                 default "same"
                                         filtered=True,        # whether to filter out unsupported characters
                                         character="abcdefg"   # support character list
                                      )
            label_start_index(list[int]):When gt_labels are labeled in `str` type, we will transfer them into `int` type
                                     according to `classes_config`. The start label will be added. e.g., for mmdet 1.x,
                                     this value is set to [1];  for mmdet 2.x, this will be set to [0].
            poly2mask (boolean):      Whether to convert the instance masks from polygons to bitmaps. Default: True.
            only_quad (boolean): Whether only quad format annotation supported.
        """
        self.with_bbox = with_bbox
        self.with_poly_bbox = with_poly_bbox
        self.with_poly_mask = with_poly_mask
        self.with_care = with_care
        self.with_label = with_label
        self.with_multi_label = with_multi_label
        self.with_text = with_text
        self.bieo_labels = bieo_labels
        self.text_profile = text_profile
        self.label_start_index = label_start_index
        self.with_cbbox = with_cbbox
        self.poly2mask = poly2mask
        self.only_quad = only_quad

        assert not (self.with_label and self.with_multi_label), \
            "Only one of with_label and with_multi_label can be true"

        if self.bieo_labels is not None:
            self.with_box_bieo_labels = self.bieo_labels['with_box_bieo_labels']
            self.pad_to_max_length = self.bieo_labels['pad_to_max_length']
            self.max_length = self.bieo_labels['max_length']
            self.padding_idx = self.bieo_labels['padding_idx']

        # Loading character dictionary
        if text_profile is not None:
            if 'character' in self.text_profile:
                # If the characters are listed in a file
                if osp.exists(self.text_profile['character']):
                    print("loading characters from file: %s" % self.text_profile['character'])
                    with open(self.text_profile['character'], 'r', encoding='utf8') as character_file:
                        characters = character_file.readline().strip().split(' ')
                        self.character = ''.join(characters)
                # If the characters are concatenated into a str.
                elif isinstance(self.text_profile['character'], str):
                    self.character = self.text_profile['character']
                else:
                    raise NotImplementedError
            else:
                self.character = ''

            # Maximum supported text length
            if 'text_max_length' in self.text_profile:
                self.text_max_length = self.text_profile['text_max_length']
            else:
                self.text_max_length = 25

            # Whether to transfer all characters into 'upper' format/ 'lower' format/ 'same' (no transfer)
            if 'sensitive' in self.text_profile:
                self.sensitive = self.text_profile['sensitive']
                if self.sensitive not in ['upper', 'lower', 'same']:
                    self.sensitive = 'same'
                    warnings.warn(
                        'sensitive type should be in ["lower","upper","same"], but found {}'
                        ' other inputs will be treated as "same" automatically'.format(self.sensitive))
            else:
                self.sensitive = 'same'

            # Whether to filter out unsupported characters (not in dictionary)
            if 'filtered' in self.text_profile:
                self.filtered = self.text_profile['filtered']
            else:
                self.filtered = True

    def _load_cares(self, results):
        """ Load and parse results['cares']

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['cares'] will be updated.
        """
        ann = results['ann_info']

        # If there is no 'bboxes' in annotations, the length is set to 1.
        bboxes_length = len(ann['bboxes']) if 'bboxes' in ann else 1

        # If there is no 'cares' in annotations or 'with_cares=False'. the results["cares"] will be the list of 1.
        if self.with_care:
            cares = np.array(ann.get('cares', np.ones(bboxes_length)))
        else:
            cares = np.ones(bboxes_length)

        results["cares"] = cares
        return results

    def _load_bboxes(self, results):
        """ Load and parse results['bboxes'], bboxes are labeled in form of axis-aligned bounding boxes, e.g.,
            [[x_min, y_min, x_max, y_max],...].

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['bboxes'] and results['bboxes_ignore'] will be updated.
        """
        ann = results['ann_info']
        cares = results['cares']

        tmp_gt_bboxes = ann.get('bboxes', [])

        gt_bboxes = []
        gt_bboxes_ignore = []
        for i, box in enumerate(tmp_gt_bboxes):
            box_i = np.array(box, dtype=np.double)
            x_coords = box_i[0::2]
            y_coords = box_i[1::2]
            aligned_box = [
                np.min(x_coords),
                np.min(y_coords),
                np.max(x_coords),
                np.max(y_coords)
            ]
            if cares[i] == 1:
                gt_bboxes.append(aligned_box)
            else:
                gt_bboxes_ignore.append(aligned_box)

        # If there is no bboxes in an image, we fill the results with a np array in shape of (0, 4)
        if len(gt_bboxes) == 0:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)

        if len(gt_bboxes_ignore) == 0:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        else:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_ignore'] = gt_bboxes_ignore

        results['bbox_fields'].append('gt_bboxes')
        results['bbox_fields'].append('gt_bboxes_ignore')
        return results

    def _load_poly_bboxes(self, results):
        """ Load and parse results['bboxes'], bboxes are labeled in form of polygon bounding boxes, e.g.,
            [[x1, y1, x2, y2,...,xn,yn],...].

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['poly_bboxes'] and results['poly_bboxes_ignore'] will be updated.
        """
        ann = results['ann_info']
        cares = results['cares']
        tmp_gt_bboxes = ann.get('bboxes', [])
        gt_poly_bboxes = []
        gt_poly_bboxes_ignore = []

        height, width = results['img_info']['height'], results['img_info']['width']

        for i, box in enumerate(tmp_gt_bboxes):
            for cor_idx in range(0, len(box), 2):
                box[cor_idx] = min(max(0, box[cor_idx]), width)
                box[cor_idx + 1] = min(max(0, box[cor_idx + 1]), height)

            # If the bboxes are labeled in 2-point form, then transfer it into 4-point form.
            if len(box) == 4:
                box = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]

            if self.only_quad and len(box) != 8:
                continue

            if self.only_quad:
                box = self.sorted_bbox_convex(box.copy())
                if not self.is_convex(box.copy()):
                    continue

            if cares[i] == 1:
                gt_poly_bboxes.append(np.array(box))
            else:
                gt_poly_bboxes_ignore.append(np.array(box))

        # If there is no bboxes in an image, we fill the results with a np array in shape of (0, 8)
        if len(gt_poly_bboxes) == 0:
            gt_poly_bboxes = np.zeros((0, 8), dtype=np.float32)

        if len(gt_poly_bboxes_ignore) == 0:
            gt_poly_bboxes_ignore = np.zeros((0, 8), dtype=np.float32)

        results['gt_poly_bboxes'] = gt_poly_bboxes
        results['gt_poly_bboxes_ignore'] = gt_poly_bboxes_ignore

        results['bbox_fields'].append('gt_poly_bboxes')
        results['bbox_fields'].append('gt_poly_bboxes_ignore')
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to bitmaps. Refer to mmdet

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons. Refer to mmdet.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[np.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def is_convex(self, bbox, area=2):
        """ Determine if a quadrilateral is a convex polygon

        Args:
            bbox (list[float]): coordinate
            area (int): minimum area

        Returns:
            bool: whether a convex polygon
        """
        pre = 1
        n = 8
        for i in range(n // 2):
            cur = (bbox[(i * 2 + 2) % n] - bbox[i * 2]) * (bbox[(i * 2 + 5) % n] - bbox[(i * 2 + 3) % n]) \
                    - (bbox[(i * 2 + 4) % n] - bbox[(i * 2 + 2) % n])\
                    * (bbox[(i * 2 + 3) % n] - bbox[(i * 2 + 1) % n])
            if cur < area:
                return False
            else:
                if cur * pre < 0:
                    return False
                else:
                    pre = cur
        return True

    def sorted_bbox_convex(self, bbox):
        """ 
        Args:
            bbox (list[float]): coordinate

        Returns:
            list[float]: sorted bbox
        """
        assert len(bbox) == 8

        bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]], [bbox[6], bbox[7]]]
        tmp_bbox = bbox.copy()
        tmp_bbox = sorted(tmp_bbox, key=lambda x: x[0])
        new_bbox = []

        if tmp_bbox[0][1] < tmp_bbox[1][1]:
            new_bbox.append(tmp_bbox[0])
            tmp_bbox.pop(0)
        else:
            new_bbox.append(tmp_bbox[1])
            tmp_bbox.pop(1)

        tmp_bbox = sorted(tmp_bbox, key=lambda x: x[1])
        for idx in range(len(tmp_bbox)):
            if tmp_bbox[idx][0] > new_bbox[0][0]:
                new_bbox.append(tmp_bbox[idx])
                tmp_bbox.pop(idx)
                break
        
        tmp_bbox = sorted(tmp_bbox, key=lambda x: x[0], reverse=True)
        new_bbox.append(tmp_bbox[0])
        new_bbox.append(tmp_bbox[1])
        
        new_bbox = [i for cor in new_bbox for i in cor]
        return new_bbox

    def _load_polymasks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`DavarCustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations. If ``self.poly2mask`` is set ``True``,
                  `gt_mask` will contain:obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        height, width = results['img_info']['height'], results['img_info']['width']

        ann = results['ann_info']
        cares = results["cares"]
        polygons = ann.get('bboxes', [])
        valid_polygons = []
        invalid_polygons = []

        for i, box in enumerate(polygons):
            for cor_idx in range(0, len(box), 2):
                box[cor_idx] = min(max(0, box[cor_idx]), width)
                box[cor_idx + 1] = min(max(0, box[cor_idx + 1]), height)

            # If the bboxes are labeled in 2-point form, then transfer it into 4-point form.
            if len(box) == 4:
                box = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]

            if self.only_quad and len(box) != 8:
                continue

            if self.only_quad:
                box = self.sorted_bbox_convex(box.copy())
                if not self.is_convex(box.copy()):
                    continue

            if cares[i] == 1:
                valid_polygons.append([np.array(box)])
            else:
                invalid_polygons.append([np.array(box)])

        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, height, width) for mask in valid_polygons], height, width)
            gt_masks_ignore = BitmapMasks(
                [self._poly2mask(mask, height, width) for mask in invalid_polygons], height, width)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in valid_polygons], height, width)
            gt_masks_ignore = PolygonMasks(
                [self.process_polygons(polygons) for polygons in invalid_polygons], height, width)

        results['gt_masks'] = gt_masks
        results['gt_masks_ignore'] = gt_masks_ignore

        results['mask_fields'].append('gt_masks')
        results['mask_fields'].append('gt_masks_ignore')
        return results

    def _load_labels(self, results):
        """ Load and parse results['labels']. Labels can be annotated in `int` or `string` format,
            e.g., [[1],[2],[3],[2],[1], ..] or [['apple'],['banana'],['orange'],['banana'],['apple'],... ].
            The first label of each bboxes will be chosen as the target label.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['gt_labels'] will be updated, in shape of [1, 2, 3, 2, 1, ...].
        """
        ann = results['ann_info']
        cares = results['cares']
        tmp_labels = ann.get("labels", None)
        bboxes_length = len(ann['bboxes']) if 'bboxes' in ann else 1

        # Select the first index
        if isinstance(self.label_start_index, list):
            self.label_start_index = self.label_start_index[0]

        # If there is no `labels` in annotation, set `label_start_index` as the default value for all bboxes.
        if tmp_labels is None:
            tmp_labels = [[self.label_start_index]] * bboxes_length
        # If `labels` in annotation are empty, set `label_start_index` as the default value for all bboxes.
        elif len(tmp_labels) == 0:
            tmp_labels = [[self.label_start_index]] * bboxes_length

        gt_labels = []
        gt_labels_ignore = []

        # Split labels according to CARE label.
        for i, label in enumerate(tmp_labels):
            if cares[i] == 1:
                gt_labels.append(label[0])
            else:
                gt_labels_ignore.append(label[0])

        # If `labels` are in string type, we transfer them into integers, and then add it with the start index.
        if len(gt_labels) > 0 and isinstance(gt_labels[0], str):
            assert results['classes_config'] is not None
            assert 'classes' in results['classes_config'] or 'classes_0' in results['classes_config']
            for i, _ in enumerate(gt_labels):
                if 'classes_0' in results['classes_config']:
                    classes_config = results['classes_config']["classes_0"]
                else:
                    classes_config = results['classes_config']['classes']

                if self.label_start_index == -1:
                    # When label_start_index is -1, means `notLabel` or `NotLabeled` is the
                    # first class in classes_config.
                    assert ('NotLabeled' in classes_config) or ('NoLabel' in classes_config)
                    if 'NotLabeled' in classes_config:
                        notlabeled_index = classes_config.index('NotLabeled')
                    else:
                        notlabeled_index = classes_config.index('NoLabel')

                    # Swap the position between NotLabel and -1
                    if notlabeled_index > 0:
                        classes_config[notlabeled_index], classes_config[0] = \
                            classes_config[0], classes_config[notlabeled_index]

                # Add labels with start_index
                gt_labels[i] = classes_config.index(gt_labels[i]) + self.label_start_index

        # If `labels_ignore` are in string type, we transfer them into integers, and then add it with the start index.
        if len(gt_labels_ignore) > 0 and isinstance(gt_labels_ignore[0], str):
            assert results['classes_config'] is not None
            assert 'classes' in results['classes_config'] or 'classes_0' in results['classes_config']
            for i, _ in enumerate(gt_labels_ignore):
                if 'classes_0' in results['classes_config']:
                    classes_config = results['classes_config']["classes_0"]
                else:
                    classes_config = results['classes_config']['classes']

                if self.label_start_index == -1:
                    # When label_start_index is -1, means `notLabel` or `NotLabeled` is the
                    # first class in classes_config.
                    assert ('NotLabeled' in classes_config) or ('NoLabel' in classes_config)
                    if 'NotLabeled' in classes_config:
                        notlabeled_index = classes_config.index('NotLabeled')
                    else:
                        notlabeled_index = classes_config.index('NoLabel')

                    # Swap the position between NotLabel and -1
                    if notlabeled_index > 0:
                        classes_config[notlabeled_index], classes_config[0] = \
                            classes_config[0], classes_config[notlabeled_index]

                # Add labels_ignore with start_index
                gt_labels_ignore[i] = classes_config.index(gt_labels_ignore[i]) + self.label_start_index

        results['gt_labels'] = gt_labels
        results['gt_labels_ignore'] = gt_labels_ignore
        #print(results['gt_labels'])
        return results

    def _load_multi_labels(self, results):
        """ Load and parse results['labels'] in multi-tasks situations. Labels can be annotated in `int` or
            `string` format, e.g., [[1,2],[2,2],[3,1],[2,1],[1,1], ..] or [['apple','big'],['banana','big'],['orange',
             'small'],['banana','small'],['apple','small'],... ].

        Args:
           results(dict): Data flow used in DavarCustomDataset.

        Returns:
           dict: updated data flow. results['gt_labels'] will be updated,
                 in shape of [[1,2], [2,2], [3,1], [2,1], [1,1], ...].
        """

        ann = results['ann_info']
        cares = results['cares']

        gt_labels = []
        gt_labels_ignore = []
        tmp_labels = ann.get('labels', None)
        bboxes_length = len(ann['bboxes']) if 'bboxes' in ann else 1

        # Count the task levels
        levels = 1
        if len(gt_labels) > 0:
            levels = len(gt_labels[0])

        # Transfer the form of label_start_index
        if isinstance(self.label_start_index, int):
            self.label_start_index = [self.label_start_index] * levels
        elif isinstance(self.label_start_index, list):
            if len(gt_labels) > 0:
                assert len(self.label_start_index) == levels
        else:
            raise ValueError("label_start_index can only be int or list")

        # If there is no `labels` in annotation, set `label_start_index` as the default value for all bboxes.
        if tmp_labels is None:
            tmp_labels = [[self.label_start_index[0]]] * bboxes_length

        # Split labels according to CARE label.
        for i, label in enumerate(tmp_labels):
            if cares[i] == 1:
                gt_labels.append(label)
            else:
                gt_labels_ignore.append(label)

        # If `labels` are in string type, we transfer them into integers, and then add it with the start index.
        for i, _ in enumerate(gt_labels):
            for j in range(levels):
                if isinstance(gt_labels[i][j], str):
                    assert results['classes_config'] is not None
                    if 'classes' in results['classes_config']:
                        results['classes_config']['classes_0'] = results['classes_config']['classes']

                    # Multiple class names loading
                    class_key = 'classes_%d' % j
                    assert class_key in results['classes_config']
                    classes_config = results['classes_config'][class_key]

                    if self.label_start_index[j] == -1:
                        assert ('NotLabeled' in classes_config) or ('NoLabel' in classes_config)
                        if 'NotLabeled' in classes_config:
                            notlabeled_index = classes_config.index('NotLabeled')
                        else:
                            notlabeled_index = classes_config.index('NoLabel')

                        # Swap the position between NotLabel and -1
                        if notlabeled_index > 0:
                            classes_config[notlabeled_index], classes_config[0] = \
                                classes_config[0], classes_config[notlabeled_index]

                    # Add labels with start_index
                    gt_labels[i][j] = classes_config.index(gt_labels[i][j]) + self.label_start_index[j]

        # If `labels_ignore` are in string type, we transfer them into integers, and then add it with the start index.
        for i, _ in enumerate(gt_labels_ignore):
            for j in range(levels):
                if isinstance(gt_labels_ignore[i][j], str):
                    assert results['classes_config'] is not None
                    if 'classes' in results['classes_config']:
                        results['classes_config']['classes_0'] = results['classes_config']['classes']

                    # Multiple class names loading
                    class_key = 'classes_%d' % j
                    assert class_key in results['classes_config']
                    classes_config = results['classes_config'][class_key]

                    if self.label_start_index[j] == -1:
                        assert ('NotLabeled' in classes_config) or ('NoLabel' in classes_config)
                        if 'NotLabeled' in classes_config:
                            notlabeled_index = classes_config.index('NotLabeled')
                        else:
                            notlabeled_index = classes_config.index('NoLabel')

                        # Swap the position between NotLabel and -1
                        if notlabeled_index > 0:
                            classes_config[notlabeled_index], classes_config[0] = \
                                classes_config[0], classes_config[notlabeled_index]

                    # Add labels_ignore with start_index
                    gt_labels_ignore[i][j] = classes_config.index(gt_labels_ignore[i][j]) + self.label_start_index[j]

        results['gt_labels'] = gt_labels
        results['gt_labels_ignore'] = gt_labels_ignore

        return results

    def _load_cbboxes(self, results):
        """ Load and parse character level bounding boxes from results['cbboxes'].

        Args:
           results(dict): Data flow used in DavarCustomDataset.

        Returns:
           dict: updated data flow. results['gt_cbboxes'] and results['gt_cbboxes_ignore'] will be updated,
                 in shape of 3-dimensional vectors [[[x1, y1, ..., x4, y4],[],[]], [[],[]] ...].
        """
        ann = results['ann_info']
        cares = results['cares']
        tmp_cbboxes = ann.get('cbboxes', [])
        tmp_gt_cbboxes = []
        tmp_gt_cbboxes_ignore = []
        for i, cbboxes in enumerate(tmp_cbboxes):
            if cares[i] == 1:
                tmp_gt_cbboxes.append(cbboxes)
            else:
                tmp_gt_cbboxes_ignore.append(cbboxes)

        results['gt_cbboxes'] = tmp_gt_cbboxes
        results['gt_cbboxes_ignore'] = tmp_gt_cbboxes_ignore
        results['cbbox_fields'].append('gt_cbboxes')
        results['cbbox_fields'].append('gt_cbboxes_ignore')

        return results

    def _load_texts(self, results):
        """ Load and parse character level bounding boxes from results['texts'].

        Args:
           results(dict): Data flow used in DavarCustomDataset.

        Returns:
           dict: updated data flow. results['gt_texts'] (list[str]) and results['gt_text'] (str) will be updated,

        """
        ann = results['ann_info']
        tmp_gt_texts = []
        tmp_texts = ann.get('texts', [])
        cares = results['cares']

        for i, text in enumerate(tmp_texts):
            # If filtered tag is True, then all unsupported characters will be removed
            if self.filtered:
                text = [c for c in text if c in self.character]
                text = ''.join(text)

            # Transfer text according to the sensitive tag.
            if self.sensitive == 'upper':
                text = text.upper()
            elif self.sensitive == 'lower':
                text = text.lower()

            # If the string is longer than supported max length, then set its CARE are False
            if len(text) > self.text_max_length:
                cares[i] = 0

            if cares[i] == 1:
                tmp_gt_texts.append(text)

        # Update cares
        results['cares'] = cares
        results['gt_texts'] = tmp_gt_texts

        # If there is only 1 text in the results['gt_texts'], results['gt_text'] is generated as the first
        # string (using in text recognition task
        if len(results['gt_texts']) == 1:
            results['gt_text'] = tmp_gt_texts[0]

        return results

    def _load_box_bieo_labels(self, results):
        """ Load and parse results['labels']. Labels can be annotated in `int` or `string` format,
            e.g., [[1],[2],[3],[2],[1], ..] or [['apple'],['banana'],['orange'],['banana'],['apple'],... ].
            The first label of each bboxes will be chosen as the target label.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: updated data flow. results['gt_labels'] will be updated, in shape of [1, 2, 3, 2, 1, ...].
        """
        ann = results['ann_info']
        cares = results['cares']
        tmp_labels = ann.get("bbox_bieo_labels", None)

        gt_labels = []
        for care, per_item in zip(cares, tmp_labels):
            if care == 1:
                gt_labels.append(per_item+[self.padding_idx]*(self.max_length - len(per_item)))

        results['gt_bieo_labels'] = gt_labels
        return results

    def __call__(self, results):
        """ Main process.

        Args:
            results(dict): Data flow used in DavarCustomDataset.

        Returns:
            dict: output data flow.
        """
        assert 'ann_info' in results
        # Cares will be used for all tasks.
        results = self._load_cares(results)

        # Loading text might affect the value of `gt_cares`
        if self.with_text:
            results = self._load_texts(results)
        if self.with_poly_bbox:
            results = self._load_poly_bboxes(results)
        if self.with_bbox:
            results = self._load_bboxes(results)
        if self.with_poly_mask:
            results = self._load_polymasks(results)
        if self.with_label:
            results = self._load_labels(results)
        if self.with_multi_label:
            results = self._load_multi_labels(results)
        if self.with_cbbox:
            results = self._load_cbboxes(results)
        if self.bieo_labels is not None and self.with_box_bieo_labels:
            results = self._load_box_bieo_labels(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            '(with_bbox={}, with_poly_bbox={}, with_poly_mask={},with_care={}, with_label={}, '
            'with_multi_lalbel={}, with_text={}, with_cbbox={}, text_profile={}, label_start_index={}, '
            'poly2mask={}, with_box_bieo_labels={}').format( self.with_bbox, self.with_poly_bbox, self.with_poly_mask, self.with_care,
                                    self.with_label, self.with_multi_label, self.with_text, self.with_cbbox,
                                    self.text_profile, self.label_start_index, self.poly2mask, self.with_box_bieo_labels)
        return repr_str
