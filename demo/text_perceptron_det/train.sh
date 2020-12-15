#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

python /path/to/mmdetection/tools/train.py  ./config/tp_r50_3stages_enlarge.py --gpus 8