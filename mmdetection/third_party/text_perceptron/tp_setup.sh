g++ -shared -o ./mmdet/datasets/pipelines/lib/tp_data.so -fPIC mmdet/datasets/pipelines/lib/tp_data.cpp `pkg-config --cflags --libs opencv`

g++ -shared -o ./mmdet/models/shape_transform_module/lib/tp_points_generate.so -fPIC mmdet/models/shape_transform_module/lib/tp_points_generate.cpp `pkg-config --cflags --libs opencv`
