#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH
DAVAROCR_PATH=/path/to/Davar-Lab-OCR/

python $DAVAROCR_PATH/davarocr/davar_videotext/tools/rcg_test.py ./config/config_gt_score.py
