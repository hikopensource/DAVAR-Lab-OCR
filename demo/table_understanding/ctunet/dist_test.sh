#!/bin/bash
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN:zh:en_US:en
export PATH=/usr/local/miniconda3/bin/:$PATH

port=`shuf -i 29500-29600 -n1`
res=`lsof -i:${port}`
while [[ -n ${res} ]]; do
    port=$((port + 1))
    res=`lsof -i:${port}`
done

PORT=${PORT:-${port}}

echo $PORT

DAVAROCR_PATH=/path/to/DAVAROCR
python -m torch.distributed.launch --nproc_per_node 1 --master_port=$PORT $DAVAROCR_PATH/tools/test.py ./configs/ctunet_chn.py "path/to/chn_model" --eval "tree_f1" --launcher pytorch
python ./tools/evaluation.py "path/to/ground_truth.json" "path/to/prediction.json"