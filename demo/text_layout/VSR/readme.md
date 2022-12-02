# VSR

This code repository contains the implementations of the paper [VSR:  A Unified Framework for Document Layout Analysis combining Vision, Semantics and Relations](https://arxiv.org/pdf/2105.06220.pdf) (ICDAR2021).

## Dataset Preparation

The demos are conducted on two public datasets: PubLayNet and DocBank. Due to the policy, you should download the original data and annoations from the official websites.
- [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet): PubLayNet is a large
 dataset of document images, of which the layout is annotated with both bounding boxes and polygonal segmentations. To perform multimodal layout analysis task, we also need to extract the annotations in the **character** granularity, in addition to layout component granularity. We provide demo examples in `demo/text_layout/datalist/PubLayNet` and one can get the above annotations through:
- [DocBank](https://github.com/doc-analysis/DocBank): DocBank is a new large-scale dataset that is constructed using a weak supervision approach. It enables models to integrate both the textual and layout information for downstream tasks. The current DocBank dataset totally includes 500K document pages, where 400K for training, 50K for validation and 50K for testing. Please download this dataset and convert annotations to *Davar* format (please refer to `demo/text_layout/datalist/DocBank`)

Please format the datalist as the form that davarocr uses according to [instructions](../datalist/readme.md).

## Train From Scratch

If you want to re-implement the model's performance from scratch, please following these steps:

1.Firstly, prepare the pretrained models:

-  [pretrained mask-rcnn model](https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=soZRTJWw) (Access Code：U743) on COCO (*we just copy the params of backbone to initialize backbone_semantic*)
-  [bert-base-uncased](https://huggingface.co/bert-base-uncased)

2.Secondly, modify the paths in model config (`demo/text_layout/VSR/PubLayNet/config/publaynet_x101.py` or `demo/text_layout/VSR/DocBank/config/docbank_x101.py`.), including the pretrained models paths, images paths, work space, etc. 

3.Thirdly, direct run `demo/text_layout/VSR/PubLayNet/dist_train.sh` or `demo/text_layout/VSR/DocBank/dist_train.sh`.

## Test

Given the trained model, direct run `demo/text_layout/VSR/PubLayNet/test.sh` or `demo/text_layout/VSR/DocBank/test.sh` to test model.

## Trained Model Download

All of the models are re-implemented and well trained based on the opensourced framework mmdetection. So, the results might be slightly different from reported results.

Trained models can be download as follows:

|  Dataset  |  Backbone  | Pretrained |  Test Scale  |  AP   |                           Links                             |
| :-------: | :--------: | :--------: | :---------: | :---: |  :----------------------------------------------------------: |
| PubLayNet (Reported) | ResNext101 |    COCO    | (1300, 800) | 95.7  |  - |
| PubLayNet | ResNext101 |    COCO    | (1300, 800) | 95.8     | [config](./PubLayNet/config/publaynet_x101.py), [pth](https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=ZF8TQD80) (Access Code: 8Rm1) |
|  DocBank (Reported) | ResNext101 |    COCO    | (600, 800)  |  95.59      |  - |
|  DocBank  | ResNext101 |    COCO    | (600, 800)  | 95.25  | [config](./DocBank/config/docbank_x101.py), [pth](https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=vp6uWpjO) (Access Code: 6T64 ) |


## Citation

If you find this repository is helpful to your research, please feel free to cite us:

``` markdown
@inproceedings{zhang2020acmmm20,
  title={{VSR:} {A} Unified Framework for Document Layout Analysis Combining Vision, Semantics and Relations},
  author={Peng, Zhang and Can, Li and Liang, Qiao, and Zhanzhan, Cheng and Shiliang, Pu and Yi, Niu and Fei, Wu},
  booktitle={16th International Conference on Document Analysis and Recognition ({ICDAR})},
  pages={115--130},
  year={2021}
}
```

## License

This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright

If there is any suggestion and problem, please feel free to contact the author with zhangpeng23@hikvision.com, qiaoliang6@hikvision.com or chengzhanzhan@hikvision.com.