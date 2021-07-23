"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    rcg_test.py
# Abstract       :    Implementations of the Recognition test script

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import os
import sys
import json
import argparse
import importlib
import os.path as osp
import torch
import mmcv

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.datasets import build_dataloader

from davarocr.davar_common.datasets.builder import davar_build_dataset as build_dataset
from davarocr.davar_rcg.models.builder import build_recognizor
from davarocr.davar_common.apis import single_gpu_test, multi_gpu_test

from .test_utils import make_paths, show_result_table, results2json, eval_json

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
    cfg.do_vis = getattr(cfg, "do_vis", {})
    if cfg.do_vis != {}:
        cfg.do_vis = getattr(cfg.do_vis, 'flag', None)
    else:
        cfg.do_vis['flag'] = None

    cfg.do_compare = getattr(cfg, "do_compare", {})
    if cfg.do_compare != {}:
        cfg.do_compare = getattr(cfg.do_compare, 'compare', None)
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
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.data.test.test_mode = True

    # initialization the distributed environment
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # initialization parameters
    init_parameters(cfg)

    # load the recognition dictionary
    if "character" in cfg:
        if osp.isfile(cfg.character):
            with open(cfg.character, 'r', encoding='utf8') as character_file:
                if cfg.character.endswith('.json'):
                    character = json.load(character_file)
                    translate_table = dict()
                    for k, value in character['char2char'].items():
                        translate_table[ord(k)] = ord(value)
                elif cfg.character.endswith('.txt'):
                    character = character_file.readline().strip()
                    translate_table = dict()
        else:
            character = cfg.character
            translate_table = dict()
    else:
        translate_table = dict()

    # The test environment is set up and establish the output path
    testsets = cfg.testsets
    ckpts = cfg.ckpts
    make_paths(cfg.out_path)
    if cfg.do_test:
        make_paths(cfg.test_path)
    if cfg.do_eval:
        make_paths(cfg.eval_path)

    total_dataset = []
    total_result = []

    if cfg.do_test or cfg.do_eval:
        for t_index, testset in enumerate(testsets):
            # load the test dataset setting
            tset = dict(
                test_mode=cfg.data.test.test_mode,
                type=cfg.data.test.type,
                data_type=cfg.data.test.info[t_index]['Type'],
                ann_file=cfg.data.test.info[t_index]["AnnFile"],
                img_prefix=cfg.data.test.info[t_index]["FilePre"],
                batch_max_length=cfg.data.test.batch_max_length,
                used_ratio=cfg.data.test.used_ratio,
                pipeline=cfg.data.test.pipeline,
                test_filter=None,)
                # getattr(cfg.data[t_index], 'Filter', None), )
            # build the test dataset
            dataset = build_dataset(tset)
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=cfg.data.samples_per_gpu if "samples_per_gpu" in cfg.data else cfg.data.imgs_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)

            # load the test model
            if cfg.tmp_dict['Epochs'] not in total_result and cfg.tmp_dict['Epochs'] is not None:
                total_result.append(cfg.tmp_dict['Epochs'])
            if 'Best_epoch' not in total_result and cfg.tmp_dict['Epochs'] is None:
                total_result.append('Best_epoch')
            for c_index, ckpt in enumerate(ckpts):
                temp_result = []
                config_file = ckpt['ConfigPath']

                # test mode
                # 1.ckpt['Epochs']  = None  - single model test mode;
                # 2.ckpt['Epochs'] != None  - multiple models test mode

                # multiple models test mode
                if ckpt['Epochs'] is not None:
                    for epoch in ckpt['Epochs']:
                        model_path = ckpt['ModelPath'] + str(epoch) + '.pth'
                        if testset['Name'] not in total_dataset:
                            total_dataset.append(testset['Name'])
                        if not os.path.exists(model_path):
                            print(model_path + ' not exist.')
                            continue

                        print('Processing {} \n'
                              'epoch of {}/{} \n'
                              'ckpts in {}/{} \n'
                              'testsets... \n'
                              'model_name:{} \n'
                              'testset_name:{} \n'
                              '/model_path:{} \n'
                              'config_path:{}\n'.format(epoch, c_index,
                                                        len(ckpts), t_index,
                                                        len(testsets),
                                                        ckpt['Name'],
                                                        testset['Name'],
                                                        model_path,
                                                        config_file))

                        res_path = \
                            cfg.test_path + testset['Name'] + '@' + ckpt['Name'] + '_e' + str(epoch) + '_res.json'
                        eval_path = \
                            cfg.eval_path + testset['Name'] + '@' + ckpt['Name'] + '_e' + str(epoch) + '_eval.json'

                        # --------------------test-------------------
                        if cfg.do_test:
                            if os.path.exists(res_path) and not cfg.force_test:
                                print(res_path + ' already exists!')
                            else:
                                # load the model config file
                                config_cfg = mmcv.Config.fromfile(config_file)
                                test_cfg = config_cfg.test_cfg

                                # build the recognition model
                                model = build_recognizor(config_cfg.model,
                                                         train_cfg=None,
                                                         test_cfg=test_cfg)

                                # load the model pth file
                                checkpoint = load_checkpoint(model,
                                                             model_path,
                                                             map_location='cpu')

                                if 'CLASSES' in checkpoint['meta']:
                                    model.CLASSES = checkpoint['meta']['CLASSES']
                                else:
                                    model.CLASSES = dataset.CLASSES

                                # test setting, single gpu test and multiple gpu test
                                if not distributed:
                                    # single gpu test
                                    model = MMDataParallel(model, device_ids=[0])
                                    outputs = single_gpu_test(model, data_loader,
                                                              args.show, model_type="RECOGNIZOR")
                                else:
                                    # multiple gpu test
                                    model = MMDistributedDataParallel(model.cuda())
                                    outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                                             args.gpu_collect, model_type="RECOGNIZOR")
                                rank, _ = get_dist_info()
                                if rank == 0:
                                    # save the model prediction result
                                    results2json(dataset, outputs, res_path,
                                                 translate_table, testsets[t_index]['PipeLine'][c_index]["sensitive"])

                        # ----------------------- eval --------------------
                        if cfg.do_eval:
                            if os.path.exists(eval_path) and not cfg.force_eval:
                                print(eval_path + ' already exists!')
                            else:
                                # evaluation result
                                acc_result = eval_json(res_path, eval_path, cfg.data.test.batch_max_length)
                                temp_result.append(acc_result)
                    total_result.append(temp_result)

                # single model test mode
                else:
                    # model path
                    model_path = ckpt['ModelPath']
                    total_dataset.append(testset['Name'])
                    if not os.path.exists(model_path):
                        print(model_path + ' not exist.')
                        return
                    res_path = cfg.test_path + testset['Name'] + '@' + ckpt['Name'] + '_res.json'
                    eval_path = cfg.eval_path + testset['Name'] + '@' + ckpt['Name'] + '_eval.json'
                    # --------------------test-------------------
                    if cfg.do_test:
                        if os.path.exists(res_path) and not cfg.force_test:
                            print(res_path + ' already exists!')
                        else:
                            # load the model config file
                            config_cfg = mmcv.Config.fromfile(config_file)
                            print(config_file)
                            test_cfg = config_cfg.test_cfg

                            # build recognition model
                            model = build_recognizor(config_cfg.model, train_cfg=None, test_cfg=test_cfg)

                            # load the model pth file
                            checkpoint = load_checkpoint(model, model_path, map_location='cpu')

                            if 'CLASSES' in checkpoint['meta']:
                                model.CLASSES = checkpoint['meta']['CLASSES']
                            else:
                                model.CLASSES = dataset.CLASSES

                            # test setting, single gpu test and multiple gpu test
                            if not distributed:
                                # single gpu test
                                model = MMDataParallel(model, device_ids=[0])
                                outputs = single_gpu_test(model, data_loader,
                                                          args.show, model_type="RECOGNIZOR")
                            else:
                                # multiple gpu test
                                model = MMDistributedDataParallel(model.cuda())
                                outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                                         args.gpu_collect, model_type="RECOGNIZOR")

                            rank, _ = get_dist_info()
                            if rank == 0:
                                # save the model prediction result
                                results2json(dataset, outputs,
                                             res_path, translate_table,
                                             testsets[t_index]['PipeLine'][c_index]["sensitive"])  # save result

                    # ----------------------- eval --------------------
                    if cfg.do_eval:
                        if os.path.exists(eval_path) and not cfg.force_eval:
                            print(eval_path + ' already exists!')
                        else:
                            # evaluation result
                            acc_result = eval_json(res_path, eval_path, cfg.data.test.batch_max_length)
                            total_result.append(acc_result)

        show_result_table(total_dataset, total_result)


if __name__ == '__main__':
    main()
