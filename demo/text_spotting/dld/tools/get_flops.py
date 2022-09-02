"""
#################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    get_flops.py
# Abstract       :    Calculate model flops

# Current Version:    1.0.0
# Date           :    2022-07-07
#################################################################################################
"""
import torch
import time
import json
from tqdm import tqdm
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from davarocr.davar_common.apis import init_model
from fvcore.nn import FlopCountAnalysis

config_file = '../configs/mask_rcnn_distill.py'
checkpoint_file = '../log/checkpoint/mask_rcnn_res50_distill_y_0.1-4a9000b6.pth'  # Model weights

model = init_model(config_file, checkpoint_file, device='cuda:0')

test_dataset = '../../datalist/total_text_test_datalist.json'
img_prefix = '/path/to/Total-Text/'

cfg = model.cfg
device = next(model.parameters()).device
test_pipeline = Compose(cfg.data.test.pipeline)
device = int(str(device).split(":")[-1])

whole_time = 0
with open(test_dataset) as load_f:
    test_file = json.load(load_f, encoding="utf-8" )

whole_flops = 0
model.forward = model.forward_dummy
for filename, _ in tqdm(test_file.items()):
    img_path = img_prefix + filename
    data = dict(img=img_path)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    with torch.no_grad():
        flops = FlopCountAnalysis(model, data['img'][0])
        whole_flops += flops.total()
whole_flops = whole_flops / (1024 ** 3)
print("flops:", whole_flops / len(test_file))
