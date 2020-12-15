#!/bin/bash
ROOT=$(cd $(dirname $0) && pwd )
echo $ROOT
PYTHON=${PYTHON:-"python"}
PIP=${PIP:-"pip"}

# install the dependencies
#$PIP install torch==1.3.0 torchvision==0.4.1
$PIP install addict six pytest pytest-runner terminaltables numpy matplotlib cython requests pyyaml Cython opencv-python prefetch_generator nltk lmdb natsort Polygon3 scikit-image Shapely albumentations==0.3.2 imagecorruptions Pillow==6.2.2 imgaug==0.3.0  onnx SharedArray tqdm pyclipper
$PIP install future tensorboard

###### install mmcv ######
cd $ROOT/mmcv/
$PYTHON setup.py develop 


###### install mmdetection ######
# When install mmdetection, we must firstly comment the 'from third_party import *' in mmdet/__init__.py, else it will cause 'cannot import deform_conv_cuda' error

cd $ROOT
sed -i 's/from third_party import/#from third_party import/g' mmdetection/mmdet/__init__.py
cd $ROOT/mmdetection/
$PYTHON setup.py build
$PYTHON setup.py develop
cd $ROOT
sed -i 's/#from third_party import/from third_party import/g' mmdetection/mmdet/__init__.py
