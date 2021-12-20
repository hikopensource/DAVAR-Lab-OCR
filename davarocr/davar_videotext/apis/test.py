"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test.py
# Abstract       :    The common testing api for video text recognition, track, quality score

# Current Version:    1.0.0
# Date           :    2021-06-02
##################################################################################################
"""
import numpy as np

import mmcv
import torch


def single_gpu_test(model,
                    data_loader):
    """ Test model with single GPU, used for visualization.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        dict: test results
    """

    model.eval()
    results = dict()
    results['texts'] = []
    results['img_info'] = []
    results['glimpses'] = []
    results['scores'] = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        texts = result['text']
        glimpses = result['glimpses']
        glimpses = glimpses.cpu().numpy()
        img_infos = result['img_info']
        scores = result['scores']
        scores = scores.cpu().numpy()
        scores = scores.reshape(-1)
        batch_size = len(texts)
        results['texts'].extend(texts)
        results['img_info'].extend(img_infos)
        results['glimpses'].extend(glimpses)
        results['scores'].extend(scores)
        for _ in range(batch_size):
            prog_bar.update()
    new_glimpse = np.stack(results['glimpses'])
    results['glimpses'] = new_glimpse
    return results
