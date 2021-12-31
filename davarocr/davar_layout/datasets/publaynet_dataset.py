"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    publaynet_dataset.py
# Abstract       :    Dataset definition for publaynet dataset.

# Current Version:    1.0.0
# Date           :    2020-12-06
##################################################################################################
"""
import json
import copy
import tempfile
import itertools
import os.path as osp
from collections import OrderedDict
import numpy as np
from terminaltables import AsciiTable

from mmdet.datasets.builder import DATASETS
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import mmcv

from .mm_layout_dataset import MMLayoutDataset


@DATASETS.register_module()
class PublaynetDataset(MMLayoutDataset):
    """
    Dataset defination for publaynet dataset.

    Ref: [1] PubLayNet: largest dataset ever for document layout analysis.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 classes_config=None,
                 ann_prefix='',
                 classes=None,
                 eval_level=1,
                 coco_ann=None):
        """
        Args:
            ann_file(str): the path to datalist.
            pipeline(list(dict)): the data-flow handling pipeline
            data_root(str): the root path of the dataset
            img_prefix(str): the image prefixes
            seg_prefix(str): the segmentation maps prefixes
            proposal_file(str): the path to the preset proposal files.
            test_mode(boolean): whether in test mode
            filter_empty_gt(boolean): whether to filter out image without ground-truthes.
            classes_config(str): the path to classes config file, used to transfer 'str' labels into 'int'
            classes(str): Dataset class, default None.
            ann_prefix(str): Annotation prefix path for each annotation file.
            eval_level(int): evaluation in which level. 1 for highest level, 0 for lowest level.
            coco_ann(str): path for coco annotation file.
        """
        self.coco_ann = coco_ann
        if coco_ann is not None:
            self.cocoGt = COCO(self.coco_ann)
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            classes_config=classes_config,
            ann_prefix=ann_prefix,
            classes=classes,
            eval_level=eval_level
        )

    def get_iou(self, pred_bbox, gt_bbox):
        """Calculate iou between pred_bbox and gt_bbox.

        Args:
            pred_bbox(list): coordinates of pred_bbox
            gt_bbox(list): coordinates of gt_bbox

        Return:
            int: iou between pred_bbox and gt_bbox.
        """
        pred_bbox = [pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]]
        gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]
        ixmin = max(pred_bbox[0], gt_bbox[0])
        iymin = max(pred_bbox[1], gt_bbox[1])
        ixmax = min(pred_bbox[2], gt_bbox[2])
        iymax = min(pred_bbox[3], gt_bbox[3])
        _iw = np.maximum(ixmax - ixmin + 1.0, 0.)
        _ih = np.maximum(iymax - iymin + 1.0, 0.)

        inters = _iw * _ih
        uni = (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0) + \
              (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0) - inters

        iou = inters / uni
        return iou

    def insert_annos(self, tmp_img_info):
        """Since some texts are not annotated, we insert annotations online using some rules.

        Args:
        	tmp_img_info(dict): img_info dict.

        Returns:
        	dict: updated img_info.

        """
        line_texts = tmp_img_info['ann']['texts']

        cooresp = -1
        received = -1

        # 'correspondence should be addressed to' / 'received' / 'accepted' appear in the same line?
        idx = 0
        while idx < len(line_texts) - 1:
            now_text = line_texts[idx]
            next_text = line_texts[idx + 1]
            if 'correspondence should be addressed to' in now_text.lower() and 'received' in next_text.lower() and \
                    'accepted' in next_text.lower():
                cooresp = idx
                received = idx + 1
                break
            idx += 1

        # if such lines exist and not in annotations, insert to annotations.
        if cooresp != -1 and received != -1:
            line_bboxes = tmp_img_info['ann']['bboxes']
            layout_bboxes = tmp_img_info['ann2']['bboxes']
            exists = False
            for per_bbox in layout_bboxes:
                if self.get_iou(line_bboxes[cooresp], per_bbox) > 0.5 or \
                        self.get_iou(line_bboxes[received], per_bbox) > 0.5:
                    exists = True
                    break

            if not exists:
                tmp_img_info['ann2']['bboxes'].append(line_bboxes[cooresp])
                tmp_img_info['ann2']['bboxes'].append(line_bboxes[received])

                tmp_img_info['ann2']['cares'].append(1)
                tmp_img_info['ann2']['cares'].append(1)

                tmp_img_info['ann2']['labels'].append([1])
                tmp_img_info['ann2']['labels'].append([1])

                w_0, h_0, w_1, h_1 = line_bboxes[cooresp]
                tmp_img_info['ann2']['segboxes'].append([[w_0, h_0, w_1, h_0, w_1, h_1, w_0, h_1]])
                w_0, h_0, w_1, h_1 = line_bboxes[received]
                tmp_img_info['ann2']['segboxes'].append([[w_0, h_0, w_1, h_0, w_1, h_1, w_0, h_1]])
        return tmp_img_info

    def pre_prepare(self, img_info):
        """Load per annotation file and reset img_info ann& ann2 fields.

        Args:
        	img_info(dict): img_info dict.

        Returns:
        	dict: updated img_info.

        """
        if img_info['url'] is not None:
            tmp_img_info = copy.deepcopy(img_info)
            ann = json.load(open(osp.join(self.ann_prefix, tmp_img_info['url']), 'r', encoding='utf8'))

            tmp_img_info["ann"] = ann.get("content_ann", None)

            if "content_ann2" in ann.keys():
                tmp_img_info["ann2"] = ann["content_ann2"]

                # filter non-valid annotations.
                cares = ann["content_ann2"]["cares"]
                bboxes = ann["content_ann2"]["bboxes"]
                for idx, per_bbox in enumerate(bboxes):
                    w_s, h_s, w_e, h_e = per_bbox
                    if w_e > w_s and h_e > h_s:
                        continue
                    else:
                        cares[idx] = 0

                tmp_img_info["ann2"]["cares"] = cares

            else:
                tmp_img_info["ann2"] = None

            # insert new annotations
            return self.insert_annos(tmp_img_info)
        else:
            return img_info

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = label
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = label
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = label
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        assert isinstance(results, list), 'results must be a list'
        assert self.coco_ann is not None
        coco_ann = json.load(open(self.coco_ann, 'r', encoding='utf8'))['images']
        coco_name_2_id = {per['file_name']: per['id'] for per in coco_ann}
        self.img_ids = [coco_name_2_id[osp.basename(per['filename'])] for per in self.data_infos]

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 jsonfile_prefix=None,
                 classwise=True,
                 metric_items=None,
                 **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        format_result, tmp_dir = self.format_results(results, jsonfile_prefix=jsonfile_prefix, **kwargs)

        # evaluate
        eval_results = OrderedDict()

        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')

        for metric in metrics:
            print('Evaluating {}...'.format(metric))
            cocoDt = self.cocoGt.loadRes(format_result[metric])

            # only evaluate on valid imgids
            imgIds = self.img_ids

            # using cocoeval API
            cocoEval = COCOeval(self.cocoGt, cocoDt, metric)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            # print classwise information
            if classwise:

                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']

                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.CLASSES) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.CLASSES):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = catId
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (nm, f'{float(ap):0.3f}'))

                # format result for showing
                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print('\n' + table.table)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
