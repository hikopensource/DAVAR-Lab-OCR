"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    inference.py
# Abstract       :    The common inference api for davarocr used in offline testing.
                       Support for DETECTOR, RECOGNIZOR, SPOTTER, INFO_EXTRACTOR, etc.

# Current Version:    1.0.0
# Date           :    2021-05-20
##################################################################################################
"""
import warnings
import torch
import numpy as np
import mmcv

from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.core import get_classes


def init_model(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a model from config file.

    Model types can be 'DETECTOR'(default), 'RECOGNIZOR', 'SPOTTER', 'INFO_EXTRACTOR'

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None

    # Can be extended according to the supported model types
    cfg_types = config.get("type", "DETECTOR")
    if cfg_types == "DETECTOR":
        model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    elif cfg_types == "RECOGNIZOR":
        from davarocr.davar_rcg.models.builder import build_recognizor
        model = build_recognizor(config.model, test_cfg=config.get('test_cfg'))
    elif cfg_types == "SPOTTER":
        from davarocr.davar_spotting.models.builder import build_spotter
        model = build_spotter(config.model, test_cfg=config.get('test_cfg'))
    elif cfg_types == "NER":
        from davarocr.davar_ner.models.builder import build_ner
        model = build_ner(config.model, test_cfg=config.get('test_cfg'))
    else:
        raise NotImplementedError

    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')

    # Save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_model(model, imgs):
    """ Inference image(s) with the models
        Model types can be 'DETECTOR'(default), 'RECOGNIZOR', 'SPOTTER', 'INFO_EXTRACTOR'

    Args:
        model (nn.Module): The loaded model
        imgs (str | nd.array | list(str|nd.array)): Image files. It can be a filename of np array (single img inference)
                                                    or a list of filenames | np.array (batch imgs inference.

    Returns:
        result (dict): results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # Build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)

    # Prepare data
    if isinstance(imgs, dict):
        data = imgs
        data = test_pipeline(data)
        device = int(str(device).split(":")[-1])
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    elif isinstance(imgs, (str, np.ndarray)):
        # If the input is single image
        data = dict(img=imgs)
        data = test_pipeline(data)
        device = int(str(device).split(":")[-1])
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    else:
        # If the input are batch of images
        batch_data = []
        for img in imgs:
            if isinstance(img, dict):
                data = dict(img_info=img)
            else:
                data = dict(img=img)
            data = test_pipeline(data)
            batch_data.append(data)
        data_collate = collate(batch_data, samples_per_gpu=len(batch_data))
        device = int(str(device).rsplit(':', maxsplit=1)[-1])
        data = scatter(data_collate, [device])[0]

    # Forward inference
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result
