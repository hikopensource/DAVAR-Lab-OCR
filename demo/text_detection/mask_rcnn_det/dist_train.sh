#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

DAVAROCR_PATH=/data1/davarocr/
python -m torch.distributed.launch --nproc_per_node 8 $DAVAROCR_PATH/tools/train.py  ./config/mask_rcnn_r50_fpn.py --launcher pytorch