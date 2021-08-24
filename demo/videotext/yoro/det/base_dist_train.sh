#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

DAVAROCR_PATH=/data1/davarocr/
python -m torch.distributed.launch --nproc_per_node 2 /data1/master/davarocr/tools/train.py  /data1/master/demo/text_detection/east/config/east_r50_rbox.py --launcher pytorch