"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test.py
# Abstract       :    The common testing api for davarocr, used in online/offline validation
                       Support for DETECTOR, RECOGNIZOR, SPOTTER, INFO_EXTRACTOR, etc.

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
import os.path as osp

import time

import mmcv
import torch

from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from mmdet.apis.test import collect_results_cpu, collect_results_gpu

from davarocr.mmcv import DavarProgressBar


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    model_type="DETECTOR",
                    min_time_interval=1):
    """ Test model with single GPU, used for visualization.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (boolean): whether to show visualization
        out_dir (str): visualization results saved path
        show_score_thr (float): the threshold to show visualization.
        model_type(str): model type indicator, used to formalize final results.
        min_time_interval(int): progressbar minimal update unit
    Returns:
        dict: test results
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = DavarProgressBar(len(dataset), min_time_interval=min_time_interval)
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)

        # Format results according to different model types
        if model_type == "DETECTOR":
            if show or out_dir:
                if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    height, width, _ = img_meta['img_shape']
                    img_show = img[:height, :width, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)

            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        elif model_type == "RECOGNIZOR":
            if "prob" in result:
                result = result["text"]
            elif "length" in result and "text" not in result:
                result = result["length"]
            elif "length" in result and "text" in result:
                result = list(zip(result["text"], result["length"]))
            else:
                result = result["text"]
            batch_size = len(result) if not isinstance(result[0], list) else len(result[0])
        elif model_type == "SPOTTER":
            pass
            # if isinstance(result[0], dict):
            #     # Remove useless key
            #     useless_keys = []
            #     for res in result:
            #         for key in res.keys():
            #             if key not in ['points', 'texts', 'confidence']:
            #                 useless_keys.append(key)
            #         for key in useless_keys:
            #             del res[key]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   model_type="DETECTOR",
                   min_time_interval=1):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        model_type(str): model type indicator, used to formalize final results.
        min_time_interval(int): progressbar minimal update unit

    Returns:
        list(dict): The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = DavarProgressBar(len(dataset), min_time_interval=min_time_interval)
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for _, data in enumerate(data_loader):

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

            # Format results according to different model types
            if model_type == "DETECTOR":
                # Encode mask results
                if isinstance(result[0], tuple):
                    result = [(bbox_results, encode_mask_results(mask_results))
                              for bbox_results, mask_results in result]
            elif model_type == "RECOGNIZOR":
                if "prob" in result:
                    result = result["text"]
                    if isinstance(result[0], list):
                        result = result[0]
                elif "length" in result and "text" not in result:
                    result = result["length"]
                elif "length" in result and "text" in result:
                    result = list(zip(result["text"], result["length"]))
                else:
                    result = result["text"]
                    if isinstance(result[0], list):
                        result = result[0]

            elif model_type == "SPOTTER":
                pass
                # if isinstance(result[0], dict):
                #     # Remove useless key
                #     useless_keys = []
                #     for res in result:
                #         for key in res.keys():
                #             if key not in ['points', 'texts', 'confidence']:
                #                 useless_keys.append(key)
                #         for key in useless_keys:
                #             del res[key]

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # Collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
