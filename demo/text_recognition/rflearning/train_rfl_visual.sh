#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

DAVAROCR_PATH=/data1/open-source/davarocr/
cd $DAVAROCR_PATH

# Stage - 1: Training the Visual Stage corresponding the configs/rfl_res32_visual.py
# Stage - 1: Use the model which is trained in Stage - 1 as the pretrained model in Stage - 3
# Stage - 3: Training the Total  Stage corresponding the configs/rfl_res32_attn.py

bash tools/dist_train.sh ../demo/text_recognition/rflearning/configs/rfl_res32_visual.py 2
