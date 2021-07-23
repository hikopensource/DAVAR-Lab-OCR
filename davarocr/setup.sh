#!/bin/bash
ROOT=$(cd $(dirname $0) && pwd )
echo $ROOT
PYTHON=${PYTHON:-"python"}
PIP=${PIP:-"pip"}

##### install davar-ocr #########
cd $ROOT
$PYTHON setup.py develop

###### Compile dependencies ######
g++ -shared -o ./davarocr/davar_det/datasets/pipelines/lib/tp_data.so -fPIC ./davarocr/davar_det/datasets/pipelines/lib/tp_data.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_det/datasets/pipelines/lib/east_data.so -fPIC ./davarocr/davar_det/datasets/pipelines/lib/east_data.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_det/core/post_processing/lib/tp_points_generate.so -fPIC ./davarocr/davar_det/core/post_processing/lib/tp_points_generate.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_det/core/post_processing/lib/east_postprocess.so -fPIC ./davarocr/davar_det/core/post_processing/lib/east_postprocess.cpp `pkg-config --cflags --libs opencv`
g++ -shared -o ./davarocr/davar_spotting/core/post_processing/lib/bfs_search.so -fPIC ./davarocr/davar_spotting/core/post_processing/lib/bfs_search.cpp `pkg-config --cflags --libs opencv`

cuda_version=$(nvcc --version | grep release | awk '{print $5}' | cut -c 1,1-2)
echo $cuda_version

if [[ $cuda_version -ge ${11} ]];then
   cd $ROOT/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/
   sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")|' CMakeLists.txt

   sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")|' CMakeLists.txt

   sed -i 's|set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")|# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")|' CMakeLists.txt
fi

cd $ROOT/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/src
rm ctc_entrypoint.cu
ln -s ctc_entrypoint.cpp ctc_entrypoint.cu


cd $ROOT/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/
mkdir build;
cd build;
cmake ..
make

cd $ROOT/davarocr/davar_rcg/third_party/warp-ctc-pytorch_bindings/pytorch_binding
$PYTHON setup.py install
