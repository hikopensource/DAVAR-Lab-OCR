mkdir ./mmdet/datasets/pipelines/lib/

g++ -shared -o ./mmdet/datasets/pipelines/lib/tp_data.so -fPIC mmdet/datasets/pipelines/src/tp_data.cpp `pkg-config --cflags --libs opencv`

mkdir ./mmdet/models/shape_transform_module/lib/

g++ -shared -o ./mmdet/models/shape_transform_module/lib/tp_points_generate.so -fPIC mmdet/models/shape_transform_module/src/tp_points_generate.cpp `pkg-config --cflags --libs opencv`
