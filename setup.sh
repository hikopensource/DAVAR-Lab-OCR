#!/bin/bash
ROOT=$(cd $(dirname $0) && pwd )
echo $ROOT
PYTHON=${PYTHON:-"python"}
PIP=${PIP:-"pip"}

# install the dependencies
# Requires torch>= 1.3.0 torchvision >= 0.4.1

# Dependencies from mmcv and mmdetection
$PIP install addict cython numpy albumentations==0.3.2 imagecorruptions matplotlib Pillow==6.2.2 six terminaltables pytest pytest-cov pytest-runner mmlvis scipy sklearn mmpycocotools yapf

# Dependencies of DavarOCR
$PIP install nltk lmdb editdistance opencv-python requests onnx SharedArray tqdm pyclipper imgaug==0.3.0 Shapely Polygon3 scikit-image prettytable transformers


###### install mmcv ######
$PIP install mmcv-full==1.3.4

###### install mmdetection #####
$PIP install mmdet==2.11.0

####### install davar-ocr ########
bash $ROOT/davarocr/setup.sh

