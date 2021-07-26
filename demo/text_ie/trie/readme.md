# TRIE

This code repository contains the implementations of the paper [TRIE: End-to-End Text Reading and Information Extraction for Document Understanding](https://arxiv.org/pdf/2005.13118.pdf) (ACM MM20).


## Preparing Dataset
- SROIE: Dataset used for [ICDAR2019-SROIE](https://rrc.cvc.uab.es/?ch=13) competition. Since this dataset contains many annotation errors, we will clean this dataset later and release corresponding trained model soon.
- Wildreceipt: Original dataset can be downloaded from [MMOCR](https://github.com/open-mmlab/mmocr). Formatted training datalist and vocab used can be found in demo/text_ie/datalist.

## Train From Scratch
Temporarily, we release pretrained model on Wildreceipt dataset using detection and recognition ground-truth, more trained models on diverse datasets will be published soon.

If you want to re-implement the model's performance from scratch, please following these steps:

1.Firstly, download original dataset and unzip it to demo/text_ie/datalist.

2.Secondly, modify the dataset, word dir and pretrained resnet50 model path in `demo/text_ie/trie/config/wildreceipt_gt_trie.py`.

3.Thirdly, direct run `demo/text_ie/trie/dist_train.sh`.

> We provide the implementation of online validation

## Test

Given the trained model, direct run `demo/text_ie/trie/test.sh` to inference model.

## Trained Model Download

All of the models are re-implemented and well trained based on the opensourced framework mmdetection. So, the results might be slightly different from reported results.

Results on various datasets and trained models can be download as follows:

|   Dataset   | Backbone | Pretrained | Train Scale | Test Scale | F1-score | Links |
| :---------: | :------: | :--------: | :---------: | :--------: | :------: | :---: |
| Wildreceipt | ResNet50 |  ImageNet  | (512, 512)  | (512, 512) |  87.08   |  [config](./configs/wildreceipt_gt_trie.py), [pth](https://pan.baidu.com/s/1TO8dtZ7HrCQrycOLAy5frg) (Access Code:parf)     |

> Note: Models are stored in BaiduYunPan, and can also be downloaded from [Google Drive](https://drive.google.com/drive/folders/1RWZJsYDYTzH7PjL48bbjBgy2BCfe0WRn?usp=sharing)


## Citation
If you find this repository is helpful to your research, please feel free to cite us:

``` markdown
@inproceedings{zhang2020acmmm20,
  title={TRIE: End-to-End Text Reading and Information Extraction for Document Understanding},
  author={Peng, Zhang and Yunlu, Xu and Zhanzhan, Cheng and Shiliang, Pu and Jing, Lu and Liang, Qiao, and Yi, Niu and Fei, Wu},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia (ACM MM)},
  pages={1413–1422},
  year={2020}
}
```
## License
This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright
If there is any suggestion and problem, please feel free to contact the author with zhangpeng23@hikvision.com or chengzhanzhan@hikvision.com.