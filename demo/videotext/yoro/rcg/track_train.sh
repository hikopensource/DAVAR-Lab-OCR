#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH
DAVAROCR_PATH=/data1/master/davarocr/
python -m torch.distributed.launch --nproc_per_node 1 /data1/master/davarocr/tools/train.py /data1/master/demo/videotext/yoro/rcg/att/configs/ic15_track_rgb_res32_bilstm_attn.py --launcher pytorch

