#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

DAVAROCR_PATH=/data1/open-source/davarocr/
cd $DAVAROCR_PATH

bash tools/dist_train.sh ../demo/text_recognition/__base__/res32_bilstm_ctc.py 2
