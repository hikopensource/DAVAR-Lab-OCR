#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

DDAVAROCR_PATH=/path/to/Davar-Lab-OCR/
python -m torch.distributed.launch --nproc_per_node 4 $DAVAROCR_PATH/tools/train.py  ./config/gcn_pn_di.py --launcher pytorch --seed 5