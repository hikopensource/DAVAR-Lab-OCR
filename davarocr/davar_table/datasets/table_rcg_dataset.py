"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    table_rcg_dataset.py
# Abstract       :    Implementation of online TEDS evaluation.

# Current Version:    1.0.0
# Date           :    2022-09-05
##################################################################################################
"""

import numpy as np
from mmdet.datasets.builder import DATASETS
from mmcv.utils import print_log
from davarocr.davar_common.datasets.davar_custom import DavarCustomDataset
from davarocr.davar_det.core.evaluation.hmean import evaluate_method
from davarocr.davar_table.core.bbox import recon_largecell
from davarocr.davar_table.utils.metric import TEDS


@DATASETS.register_module()
class TableRcgDataset(DavarCustomDataset):
    """ The format is the same as DavarCustomDataset."""

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
            ann_file(str): the path to datalist.
            pipeline(list(dict)): the data-flow handling pipeline
            data_root(str): the root path of the dataset
            img_prefix(str): the image prefixes
            seg_prefix(str): the segmentation maps prefixes
            proposal_file(str): the path to the preset proposal files.
            test_mode(boolean): whether in test mode
            filter_empty_gt(boolean): whether to filter out image without ground-truthes.
            classes_config(str): the path to classes config file, used to transfer 'str' labels into 'int'
        """
        super().__init__(ann_file, pipeline, data_root, img_prefix, seg_prefix, proposal_file, test_mode,
                         filter_empty_gt, classes_config)
        self.eval_func_params = {
            "IOU_CONSTRAINT": 0.5,  # IOU threshold for pred v.s. gt matching
            "AREA_PRECISION_CONSTRAINT": 0.5,  # IOU threshold for pred v.s. not-cared gt matching
            "CONFIDENCES": False,  # If it is True, mAP will be calculated (default False)
            "ENLARGE_ANN_BBOXES": True  # If it is True, using enlarge strategy to generate aligned cells
        }

    def evaluate(self,
                 results,
                 metric="TEDS",
                 logger=None,
                 **eval_kwargs):
        """ Main process of evaluation

        Args:
            results (list(dict)): formatted inference results,
            metric (str): default "TEDS"
            logger (obj): obj to print/ write logs
            eval_kwargs (dict): other eval kwargs.
        Returns:
            dict: evaluation results, e.g.,
                if metric is "TEDS", it looks like:
                    dict(
                          "TEDS": 0.9,
                    )
                if metric is "hmean", it looks like:
                    dict(
                          "precision": 0.9,
                          "recall": 0.9,
                          "hmean": 0.9,
                    )
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['TEDS', 'hmean']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported, only "TEST" and "hmean" are supported.')
        assert len(results) == len(self)

        output = {}
        if metric == "TEDS":
            gt_results = dict()
            pred_results = dict()
            for i in range(len(self)):
                # Prepare ground truth
                ann = self.data_infos[i]
                ann_name = ann['filename']
                ann_html = ann['ann']['html']
                gt_results[ann_name] = {'html': ann_html}

                # Prepare predictions
                pred_html = results[i]['html']
                pred_results[ann_name] = pred_html

            # evaluation
            teds = TEDS(structure_only=True, n_jobs=16)
            evaluate_result = teds.batch_evaluate(pred_results, gt_results)
            mean_score = np.array(list(evaluate_result.values())).mean()
            output['TEDS'] = mean_score
            print_log("Evaluation results: TEDS scores: {}".format(output['TEDS']), logger=logger)

        else:
            eval_func_params = eval_kwargs["eval_func_params"]
            if eval_func_params is not None and isinstance(eval_func_params, dict):
                if "IOU_CONSTRAINT" in eval_func_params:
                    self.eval_func_params["IOU_CONSTRAINT"] = eval_func_params["IOU_CONSTRAINT"]
                if "AREA_PRECISION_CONSTRAINT" in eval_func_params:
                    self.eval_func_params["AREA_PRECISION_CONSTRAINT"] = eval_func_params["AREA_PRECISION_CONSTRAINT"]
                if "CONFIDENCES" in eval_func_params:
                    self.eval_func_params["CONFIDENCES"] = eval_func_params["CONFIDENCES"]
                if "ENLARGE_ANN_BBOXES" in eval_func_params:
                    self.eval_func_params["ENLARGE_ANN_BBOXES"] = eval_func_params["ENLARGE_ANN_BBOXES"]

            det_results = []
            gt_results = []

            for i in range(len(self)):
                ann = self.get_ann_info(i)
                det_result = results[i]['content_ann']

                # Prepare predictions
                formated_det_result = dict()

                # if det_result is a 4-value list, change it into a 8-value list
                det_result['bboxes'] = [poly for poly in det_result['bboxes'] if poly]
                for j, poly in enumerate(det_result['bboxes']):
                    if len(poly) == 4:
                        tmp_box = [poly[0], poly[1], poly[2], poly[1], poly[2], poly[3], poly[0], poly[3]]
                        det_result['bboxes'][j] = tmp_box

                formated_det_result['points'] = det_result['bboxes']
                formated_det_result['confidence'] = [1.0] * len(det_result['bboxes'])
                formated_det_result["texts"] = ["*"] * len(det_result['bboxes'])

                # Prepare ground truth
                formated_gt_result = dict()
                gt_polys = ann.get('bboxes', [])
                gt_cells = ann.get('cells', [])

                # if the ENLARGE strategy should be used, use it to generate aligned cells
                if self.eval_func_params["ENLARGE_ANN_BBOXES"]:
                    # filter targets without bbox annotation
                    gt_cells = [cell for poly, cell in zip(gt_polys, gt_cells) if poly]
                    gt_polys = [poly for poly in gt_polys if poly]

                    for j, poly in enumerate(gt_polys):
                        if len(poly) == 4:
                            continue
                        elif len(poly) == 8:
                            tmp = [poly[0], poly[1], poly[4], poly[5]]
                            if poly != [tmp[0], tmp[1], tmp[2], tmp[1], tmp[2], tmp[3], tmp[0], tmp[3]]:
                                raise KeyError('The ENLARGE strategy can be used only if all cells are rectangular')
                            gt_polys[j] = tmp
                        else:
                            raise KeyError('The ENLARGE strategy can be used only if all cells are rectangular')
                    gt_polys = recon_largecell(gt_polys, gt_cells)
                else:
                    gt_polys = [poly for poly in gt_polys if poly]

                # if gt_polys is a 4-value list, change it into a 8-value list
                for j, poly in enumerate(gt_polys):
                    if len(poly) == 4:
                        tmp_box = [poly[0], poly[1], poly[2], poly[1], poly[2], poly[3], poly[0], poly[3]]
                        gt_polys[j] = tmp_box

                cares = ann.get('cares', [1] * len(gt_polys))
                gt_trans = [("*" if care == 1 else self.ignore) for care in cares]
                formated_gt_result['gt_bboxes'] = gt_polys
                formated_gt_result['gt_texts'] = gt_trans

                det_results.append(formated_det_result)
                gt_results.append(formated_gt_result)

            # evaluation
            evaluate_result = evaluate_method(det_results, gt_results, self.eval_func_params)
            output['precision'] = evaluate_result['summary']['precision']
            output['recall'] = evaluate_result['summary']['recall']
            output['hmean'] = evaluate_result['summary']['hmean']

            print_log("Evaluation results: Precision: {}, Recall: {}, hmean: {} ".format(output['precision'],
                                                                                         output['recall'],
                                                                                         output['hmean']),
                                                                                        logger=logger)

        return output
