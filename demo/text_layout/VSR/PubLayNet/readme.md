# VSR on PubLayNet

This code repository contains the implementations of the paper [VSR:  A Unified Framework for Document Layout Analysis combining Vision, Semantics and Relations](https://arxiv.org/pdf/2105.06220.pdf) (ICDAR2021).


## Preparing Dataset
- [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet): PubLayNet is a large dataset of document images, of which the layout is annotated with both bounding boxes and polygonal segmentations. To perform multimodal layout analysis task, we also need to extract the annotations in the **character** granularity, in addition to layout component granularity. We provide demo examples in `demo/text_layout/datalist/PubLayNet` and one can get the above annotations through:
  - download the PubLayNet dataset together with its PDF files
  - resort to PDF parsing tools, such as [pdfplumber](https://github.com/jsvine/pdfplumber), [pdfminer](https://github.com/pdfminer/pdfminer.six) to extract texts/ layout coordinates through PDF files.
  - convert annotations to *Davar* format (please refer to `demo/text_layout/datalist/PubLayNet` ***Datalist/ Annos/ Images***).

## Train From Scratch
If you want to re-implement the model's performance from scratch, please following these steps:

1.Firstly, download pretrained models and put them in `demo/text_layout/VSR/common`:

-  [pretrained model]() trained on COCO (*we just copy the params of backbone to initialize backbone_semantic*)
- [bert-base-uncased](https://huggingface.co/bert-base-uncased)

2.Secondly, modify the dataset, word directory and pretrained model (from step1) path in `demo/text_layout/VSR/PubLayNet/config/publaynet_x101.py`.

3.Thirdly, direct run `demo/text_layout/VSR/PubLayNet/dist_train.sh`.

> We provide the implementation of online validation

## Test

Given the trained model, direct run `demo/text_layout/VSR/PubLayNet/test.sh` to inference model.

## Trained Model Download

All of the models are re-implemented and well trained based on the opensourced framework mmdetection. So, the results might be slightly different from reported results.

Trained models can be download as follows:

|  Dataset  |  Backbone  | Pretrained |                     Train Scale                     | Test Scale  |  AP  | AP (paper) |                            Links                             |
| :-------: | :--------: | :--------: | :-------------------------------------------------: | :---------: | :--: | :--------: | :----------------------------------------------------------: |
| PubLayNet | ResNext101 |    COCO    | Multiple Scales with short size in range [640, 800] | (1300, 800) | 95.8 |    95.7    | [config](./config/publaynet_x101.py), [pth](https://pan.baidu.com/s/1TO8dtZ7HrCQrycOLAy5frg) (Access Code:parf) |


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

## Contact
If there is any suggestion and problem, please feel free to contact the author with qiaoliang6@hikvision.com.