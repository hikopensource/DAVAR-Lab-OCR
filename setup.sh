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
$PIP install nltk lmdb editdistance opencv-python requests onnx SharedArray tqdm pyclipper imgaug==0.3.0 Shapely Polygon3 scikit-image prettytable transformers seqeval Levenshtein networkx bs4 distance apted lxml jsonlines

###### install mmcv ######
$PIP install mmcv-full==1.3.4

###### install mmdetection #####
$PIP install mmdet==2.11.0

##### install davar-ocr #########
$PYTHON setup.py develop

###### Compile dependencies ######
g++ -shared -o ./davarocr/davar_det/datasets/pipelines/lib/tp_data.so -fPIC ./davarocr/davar_det/datasets/pipelines/lib/tp_data.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_det/datasets/pipelines/lib/east_data.so -fPIC ./davarocr/davar_det/datasets/pipelines/lib/east_data.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_det/core/post_processing/lib/tp_points_generate.so -fPIC ./davarocr/davar_det/core/post_processing/lib/tp_points_generate.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_det/core/post_processing/lib/east_postprocess.so -fPIC ./davarocr/davar_det/core/post_processing/lib/east_postprocess.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_spotting/core/post_processing/lib/bfs_search.so -fPIC ./davarocr/davar_spotting/core/post_processing/lib/bfs_search.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_table/datasets/pipelines/lib/gpma_data.so -fPIC ./davarocr/davar_table/datasets/pipelines/lib/gpma_data.cpp `pkg-config --cflags --libs opencv`



###### Install warpctc ######
cuda_version=$(nvcc --version | grep release | awk '{print $5}' | cut -c 1,1-2)
echo $cuda_version


if [[ $cuda_version -ge ${11} ]];then
   cd $ROOT/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/
   sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")|' CMakeLists.txt

   sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")|' CMakeLists.txt

   sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")|' CMakeLists.txt

   FIND_FILE="./CMakeLists.txt"
   FIND_STR="IF (CUDA_VERSION GREATER 9.9)"
   if [ `grep -c "$FIND_STR" $FIND_FILE` -ne '0' ];then
       echo "The File Has Ever Been Installed!"
   else
       sed -i '53s/$/\nIF (CUDA_VERSION GREATER 9.9)/g' CMakeLists.txt

       sed -i '54s/$/\n    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")/g' CMakeLists.txt

       sed -i '55s/$/\nENDIF()/g' CMakeLists.txt

       sed -i '56s/$/\n/g' CMakeLists.txt

       sed -i '57s/$/\nIF (CUDA_VERSION GREATER 10.9)/g' CMakeLists.txt

       sed -i '58s/$/\n    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")/g' CMakeLists.txt

       sed -i '59s/$/\nENDIF()/g' CMakeLists.txt
   fi
fi

cd $ROOT/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/src
rm ctc_entrypoint.cu
ln -s ctc_entrypoint.cpp ctc_entrypoint.cu


cd $ROOT/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/
mkdir -p build;
cd build;
cmake ..
make

cd $ROOT/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/pytorch_binding
$PYTHON setup.py install
