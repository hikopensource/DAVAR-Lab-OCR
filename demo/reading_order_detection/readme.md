# GCN-PN

This code repository contains the implementations of the method 

[GCN-PN]: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700086.pdf

 (ECCV2020).

## Train From Scratch

If you want to train the model from scratch, please following these steps:

1.Firstly, download the DI dataset from https://tianchi.aliyun.com/dataset/dataDetail?dataId=72926 to get 

   the Images. The corresponding datalists are in  `demo/reading_order_detection/datalist/DI/` 

2.Secondly, prepare the pretrained models:

-   *resnext101_64*x4d-ee2c6f71.pth

3.Thirdly, direct run `demo/reading_order_detection/GCN-PN/train.sh`

## Test

Given the trained model, direct run `demo/reading_order_detection/GCN-PN/test.sh` to test model.

## Trained Model Download

For the released data is a subset, which smaller than paper reported. So the results might be slightly different from reported results. Moreover, paper takes sinkhorn method into training phase and get some improvements, but it works less in our implementation. Thus, we only release the base model. 

Results on DI datasets and trained models can be download as follows:

|   total_order_acc   | DI_whole | DI_subset | Links       |
| :---------:        | :------: | :--------: | :---------: |
| GCN-PN(paper) |  79   |    -       | -   |
| GCN-PN   |  -    |   72.23   |  |



## Citation

If you find this repository is helpful to your research, please feel free to cite as:

``` markdown
@inproceedings{DBLP:conf/eccv/LiGBWYZ20,
  author    = {Liangcheng Li and
               Feiyu Gao and
               Jiajun Bu and
               Yongpan Wang and
               Zhi Yu and
               Qi Zheng},
  title     = {An End-to-End {OCR} Text Re-organization Sequence Learning for Rich-Text
               Detail Image Comprehension},
  booktitle = {Computer Vision - {ECCV} 2020 - 16th European Conference, Glasgow,
               UK, August 23-28, 2020, Proceedings, Part {XXV}},
  pages     = {85--100},
  year      = {2020},
}
```

## License

This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright

If there is any suggestion and problem, please feel free to contact the author with lican9@hikvision.com or qiaoliang6@hikvision.com.