#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH
DAVAROCR_PATH=/path/to/Davar-Lab-OCR/

python $DAVAROCR_PATH/davarocr/davar_videotext/tools/spotting_test.py $DAVAROCR_PATH/demo/videotext/yoro/rcg/att/test_scripts/test_spotting_config.py
