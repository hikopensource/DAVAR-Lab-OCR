#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

DAVAROCR_PATH=/data1/open-source/davarocr/
cd $DAVAROCR_PATH

bash tools/dist_train.sh ../demo/text_recognition/spin/configs/gaspin_res32_attn.py 2
