"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    text_spotting_dataset.py
# Abstract       :    Implementation of text spotting dataset evaluation.

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""

from mmdet.datasets.builder import DATASETS
from mmcv.utils import print_log
from davarocr.davar_common.datasets.davar_custom import DavarCustomDataset
from davarocr.davar_spotting.core.evaluation.e2e_hmean import evaluate_method


@DATASETS.register_module()
class TextSpotDataset(DavarCustomDataset):
    """ The format is the same as DavarCustomDataset. """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=False,
                 classes_config=None,
                 ):
        """
        Args:
            ann_file (str): the path to datalist.
            pipeline (list(dict)): the data-flow handling pipeline
            data_root (str): the root path of the dataset
            img_prefix (str): the image prefixes
            seg_prefix (str): the segmentation maps prefixes
            proposal_file (str): the path to the preset proposal files.
            test_mode (boolean): whether in test mode
            filter_empty_gt (boolean): whether to filter out image without ground-truthes.
            classes_config (str): the path to classes config file, used to transfer 'str' labels into 'int'

        """

        super().__init__(ann_file, pipeline, data_root, img_prefix, seg_prefix, proposal_file, test_mode,
                         filter_empty_gt, classes_config)
        self.ignore = "###"
        self.eval_func_params = {
            "IOU_CONSTRAINT": 0.5,  # IOU threshold for pred v.s. gt matching
            "AREA_PRECISION_CONSTRAINT": 0.5,  # IOU threshold for pred v.s. not-cared gt matching
            "WORD_SPOTTING": True,  # If it is True, words that not in dictionary will be not considered
            "MIN_LENGTH_CARE_WORD": 3,  # The words are shorter in length (<3) will be not considered
            "SPECIAL_CHARACTERS": "[]+-#$()@=_!?,:;/.%&\'\">*|<`{~}^\ ",
            "ONLY_REMOVE_FIRST_LAST_CHARACTER": True  # Whether to only remove the first&last character
        }

    def evaluate(self,
                 results,
                 metric="hmean",
                 logger=None,
                 **eval_kwargs):
        """ 
        Main process of evaluation.
        Args:
            results (list(dict)): formatted inference results,
                                 e.g., [{'points': [x1, y2, ..., xn,yn],
                                        'confidence':[1, 0,8,...],
                                        'texts':['apple','banana',...]},{},{},...]
            metric (str | list(str)): default "e2e_hmean"
            logger (obj): obj to print/ write logs
            **eval_kwargs: evaluation parameters, which stored in
                           eval_kwargs['eval_func_params']= dict(
                               "IOU_CONSTRAINT": 0.5 (default),
                               "AREA_PRECISION_CONSTRAINT": 0.5 (default),
                               "CONFIDENCES": FAlSE (default)).
                               
        Returns:
            dict: evaluation results, e.g.,
                  dict(
                      "precision": 0.9,
                      "recall": 0.9,
                      "hmean": 0.9,
                  )
                  
        """
        if not isinstance(metric, str):
            assert "hmean" in metric
        else:
            assert metric == "hmean"
        assert len(results) == len(self)
        eval_func_params = eval_kwargs["eval_func_params"]
        if eval_func_params is not None and isinstance(eval_func_params, dict):
            if "IOU_CONSTRAINT" in eval_func_params:
                self.eval_func_params["IOU_CONSTRAINT"] = eval_func_params["IOU_CONSTRAINT"]
            if "AREA_PRECISION_CONSTRAINT" in eval_func_params:
                self.eval_func_params["AREA_PRECISION_CONSTRAINT"] = eval_func_params["AREA_PRECISION_CONSTRAINT"]
            if "WORD_SPOTTING" in eval_func_params:
                self.eval_func_params["WORD_SPOTTING"] = eval_func_params["WORD_SPOTTING"]
            if "MIN_LENGTH_CARE_WORD" in eval_func_params:
                self.eval_func_params["MIN_LENGTH_CARE_WORD"] = eval_func_params["MIN_LENGTH_CARE_WORD"]
            if "SPECIAL_CHARACTERS" in eval_func_params:
                self.eval_func_params["SPECIAL_CHARACTERS"] = eval_func_params["SPECIAL_CHARACTERS"]
            if "ONLY_REMOVE_FIRST_LAST_CHARACTER" in eval_func_params:
                self.eval_func_params["ONLY_REMOVE_FIRST_LAST_CHARACTER"] = eval_func_params[
                    "ONLY_REMOVE_FIRST_LAST_CHARACTER"]
        print("\nDo {} evaluation with iou constraint {}...".format(
            "Word Spotting" if self.eval_func_params["WORD_SPOTTING"] else "End-to-End",
            self.eval_func_params["IOU_CONSTRAINT"]))

        det_results = []
        gt_results = []
        output = {}
        for i in range(len(self)):
            ann = self.get_ann_info(i)
            det_result = results[i]
            assert 'points' in det_result

            # Prepare predictions
            formated_det_result = dict()
            formated_det_result['points'] = det_result['points']
            formated_det_result['confidence'] = det_result['confidence'] if 'confidence' in det_result \
                else [1.0] * len(det_result['points'])
            formated_det_result["texts"] = det_result['texts'] if 'texts' in det_result \
                else ["*"] * len(det_result['points'])

            # Prepare ground truth
            formated_gt_result = dict()
            gt_polys = ann.get('bboxes', [])
            gt_texts = ann.get('texts', [])
            cares = ann.get('cares', [1] * len(gt_polys))
            assert len(cares) == len(gt_texts) == len(gt_polys)
            gt_trans = [(gt_texts[i] if care == 1 else self.ignore) for i, care in enumerate(cares)]
            formated_gt_result['gt_bboxes'] = gt_polys
            formated_gt_result['gt_texts'] = gt_trans

            det_results.append(formated_det_result)
            gt_results.append(formated_gt_result)

        evaluate_result = evaluate_method(det_results, gt_results, self.eval_func_params)
        output['precision'] = evaluate_result['summary']['spot_precision']
        output['recall'] = evaluate_result['summary']['spot_recall']
        output['hmean'] = evaluate_result['summary']['spot_hmean']

        print("Finish evaluation !")

        print_log("Detection evaluation results: Precision: {}, Recall: {}, hmean: {}".
          format(evaluate_result['summary']['det_precision'],
                 evaluate_result['summary']['det_recall'],
                 evaluate_result['summary']['det_hmean']), logger=logger)

        print_log("Spotting evaluation results: Precision: {}, Recall: {}, hmean: {}".
          format(evaluate_result['summary']['spot_precision'],
                 evaluate_result['summary']['spot_recall'],
                 evaluate_result['summary']['spot_hmean']), logger=logger)

        return output
