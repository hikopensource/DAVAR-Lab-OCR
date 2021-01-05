# Text Perceptron （detection part）

This code repository contains the implementations of the paper [Text Perceptron: Towards End-to-End Arbitrary-Shaped Text Spotting](https://arxiv.org/pdf/2002.06820.pdf) (AAAI 2020 oral).

> To date, the detection part in the Text Perceptron is re-implemented based on the common repository [Davar-OCR](https://github.com/hikopensource/DAVAR-Lab-OCR), and the recognization part will be released in the future.

The model's implementation is in `mmdetection/third_party/text_perceptron`. The running scritps/demo is in `demo/text_perceptron_det/`.

## Installation

Before running the demo code, you should compile and install corresponding dependencies in `mmdetection/third_party/`.

**Step 1**: Install Davar-OCR (complie mmdetection and mmcv) 
``` bash
>>> cd $DAVR_LAB_OCR_ROOT$
>>> bash setup.sh 
```

**Step 2**: Uncomment corresponding importing statement `from .text_perceptron import *` in `mmdetection/third_party/__init__.py`.

**Step 3**: Compile dependencies of TP model as part of our code is implemented by C++.
``` bash
>>> cd $DAVAR_LAB_OCR_ROOT$/mmdetection/third_party/text_perceptron/
>>> bash tp_setup.sh
```

## Preparing Dataset
Original images can be downloaded from: [Total-Text](https://github.com/cs-chan/Total-Text-Dataset "Total-Text") , [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector).

The formated training datalist and test datalist can be found in `demo/text_perceptron_det/datalist/`

## Training
Modified the paths of imgs/ pretrained_model/ work_space in the config files `demo/text_perceptron_det/config/tp_r50_3stages_enlarge.py`.

Run the following bash command in the command line,
``` bash
>>> cd $DAVAR_LAB_OCR_ROOT$/demo/text_perceptron_det/
>>> bash train.sh
```
## Inference
Directly modify the paths (`test_dataset`, `image_prefix`, etc.) in the testing script, and run testing:
``` bash
>>> python test.py 
```
Some visualization of detection results are shown:

![visualizations/pred_img499.jpg](visualizations/pred_img499.jpg)
![visualizations/pred_img566.jpg](visualizations/pred_img566.jpg)


## Trained Model Download

Reported results on various datasets and trained models download:

|                       | Pretrained | Precision | Recall | Hmean | Links               |
| --------------------- | ---------- | --------- | ------ | ----- | ------------------- |
| Total-Text (Reported) | SynthText  | 88.1      | 78.9   | 83.3  | -                   |
| Total-Text            | SynthText  | 85.7      | 81.4   | 83.5  | [config](config/tp_r50_3stages_enlarge.py), [pth](https://pan.baidu.com/s/1ZkccnlBvioqVrfb-g06yBQ ) (Access Code: vxzn)|
| SCUT-CTW1500 (Reported) | SynthText  | 88.7      | 78.2   | 83.1  | -                   |
| SCUT-CTW1500          | SynthText  | 86.1      | 80.0   | 82.9  | [config](config/tp_r50_3stages_enlarge.py), [pth](https://pan.baidu.com/s/1ZkccnlBvioqVrfb-g06yBQ ) (Access Code: vxzn)|
> Note: Models are stored in BaiduYunPan.

## Citation

If you find this repository is helpful to your research, please feel free to cite us:

``` markdown
@inproceedings{qiao2020text,
  title={Text Perceptron: Towards End-to-End Arbitrary-Shaped Text Spotting},
  author={Qiao, Liang and Tang, Sanli and Cheng, Zhanzhan and Xu, Yunlu and Niu, Yi and Pu, Shiliang and Wu, Fei},
  booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
  pages={11899-11907},
  year={2020}
}
```
## License
This project is released under the [Apache 2.0 license](https://github.com/hikopensource/DAVAR-Lab-OCR/blob/main/mmdetection/third_party/LICENSE)

## Copyright
If there is any suggestion and problem, please feel free to contact the author with qiaoliang6@hikvision.com or chengzhanzhan@hikvision.com.
