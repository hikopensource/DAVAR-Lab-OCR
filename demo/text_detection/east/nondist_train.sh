#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

DAVAROCR_PATH=/data1/davarocr/
python $DAVAROCR_PATH/tools/train.py --no-validate --gpus 1 ./config/east_r50_rbox.py