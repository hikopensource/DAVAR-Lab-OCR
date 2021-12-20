"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    rcg_test.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-06-10
##################################################################################################
"""

import os
import sys
import argparse
import importlib


import torch
import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import init_dist, load_checkpoint
from mmdet.datasets import build_dataloader

from davarocr.davar_common.datasets.builder import davar_build_dataset as build_dataset
from davarocr.davar_rcg.models.builder import build_recognizor
from davarocr.davar_videotext.apis import single_gpu_test


from test_utils import make_paths, gen_train_score, pred_test_score, gen_pred_text


importlib.reload(sys)


def parse_args():
    """

    Returns:
        args parameter of model test

    """
    parser = argparse.ArgumentParser(description='DavarOCR test recognition')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--json_out', help='output result file name without extension', type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def init_parameters(cfg):
    """
    Args:
        cfg (cfg): model config file

    Returns:

    """

    # parameter initialization
    cfg.do_vis = getattr(cfg, 'do_vis', {})
    if cfg.do_vis != {}:
        cfg.do_vis['flag'] = getattr(cfg.do_vis, 'flag', None)
    else:
        cfg.do_vis['flag'] = None
    cfg.do_compare = getattr(cfg, 'do_compare', {})
    if cfg.do_compare != {}:
        cfg.do_compare['compare'] = getattr(cfg.do_compare, 'compare', None)
    else:
        cfg.do_compare['compare'] = None
    cfg.do_eval_ana = getattr(cfg, 'do_eval_ana', None)
    cfg.ana_txt = getattr(cfg, 'ana_txt', None)
    cfg.ana_xls = getattr(cfg, 'ana_xls', None)


def main():
    """
    Returns:

    """
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # Set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.data.test.test_mode = True

    # Initialization the distributed environment
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # Initialization parameters
    init_parameters(cfg)

    # The test environment is set up and establish the output path
    testsets = cfg.testsets
    ckpts = cfg.ckpts
    make_paths(cfg.out_path)

    total_dataset = []
    total_result = []

    for t_index, testset in enumerate(testsets):

        # Load the test dataset setting
        tset = dict(
            test_mode=cfg.data.test.test_mode,
            type=cfg.data.test.type,
            data_type=cfg.data.test.info[t_index]['Type'],
            ann_file=cfg.data.test.info[t_index]["AnnFile"],
            img_prefix=cfg.data.test.info[t_index]["FilePre"],
            batch_max_length=cfg.data.test.batch_max_length,
            used_ratio=cfg.data.test.used_ratio,
            pipeline=cfg.data.test.pipeline,
            filter_cares=cfg.data.test.filter_cares,
            test_filter=None)

        # Build the test dataset
        dataset = build_dataset(tset)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu if "samples_per_gpu" in cfg.data else cfg.data.imgs_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # Load the test model
        if cfg.tmp_dict['Epochs'] not in total_result and cfg.tmp_dict['Epochs'] is not None:
            total_result.append(cfg.tmp_dict['Epochs'])
        if 'Best_epoch' not in total_result and cfg.tmp_dict['Epochs'] is None:
            total_result.append('Best_epoch')

        for ckpt in ckpts:
            config_file = ckpt['ConfigPath']

            # Model path
            model_path = ckpt['ModelPath']
            total_dataset.append(testset['Name'])
            if not os.path.exists(model_path):
                print(model_path + ' not exist.')
                return

            # Output file
            train_score_path = cfg.out_path + testset['Name'] + '_train_with_score.json'
            test_score_path = cfg.out_path + testset['Name'] + '_test_score.json'

            # --------------------test-------------------
            if os.path.exists(train_score_path) and not cfg.force_test:
                print(train_score_path + ' already exists!')
            else:
                # Load the model config file
                config_cfg = mmcv.Config.fromfile(config_file)
                test_cfg = config_cfg.test_cfg

                # Build recognition model
                model = build_recognizor(config_cfg.model, train_cfg=None, test_cfg=test_cfg)

                # Load the model pth file
                checkpoint = load_checkpoint(model, model_path, map_location='cpu')

                if 'CLASSES' in checkpoint['meta']:
                    model.CLASSES = checkpoint['meta']['CLASSES']
                else:
                    model.CLASSES = dataset.CLASSES

                # Single gpu test
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test(model, data_loader)

                pred_texts = outputs['texts']
                glimpses = outputs['glimpses']
                img_infos = outputs['img_info']
                scores = outputs['scores']

                # Generating train set quality scores according to cluster
                if cfg.gen_train_score:
                    gen_train_score(glimpses, img_infos,  tset['ann_file'], train_score_path)

                # Predict quality score and texts
                if cfg.pred_test_score:
                    img_infos = gen_pred_text(pred_texts, img_infos)
                    pred_test_score(scores, img_infos, test_score_path, cfg.data.test.info[t_index]["AnnFile"])


if __name__ == '__main__':
    main()
