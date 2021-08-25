# DAVAR-OCR

This is the opensourced OCR repository of [DAVAR Lab](https://davar-lab.github.io/), from Hikvision Research Institute, China. 

We begin to maintain this code repository to release the implementations of our recent academic publishments and some re-implementations of previous popular algorithms/modules in OCR. 

We also provide some of the ablation experiment comparasions for better reproduction. 

> Note: Due to the policy limits of the company. All of the codes were re-implemented based on the open-source frameworks, [mmdetection-2.11.0](https://github.com/open-mmlab/mmdetection/releases/tag/v2.11.0) and [mmcv-1.3.4](https://github.com/open-mmlab/mmcv/releases/tag/v1.3.4), from [open-mmlab](https://github.com/open-mmlab "open-mmlab"). The code architecture also refers to [mmocr](https://github.com/open-mmlab/mmocr), which means these two frameworks can be well compatible to each other.

## Implementations
To date, davarocr contains the following algorithms:

***Text Detection***

- [x] [EAST](demo/text_detection/east) (CVPR 2017)

- [x] [MASK RCNN](demo/text_detection/mask_rcnn_det) (ICCV 2017)

- [x] [Text Perceptron Det](demo/text_detection/text_perceptron_det) (AAAI 2020)

***Text Recognition***

- [x] [Attention](demo/text_recognition/__base__) (CVPR 2016)

- [x] [CRNN](demo/text_recognition/__base__) (TPAMI 2017)

- [x] [ACE](demo/text_recognition/ace) (CVPR 2019)

- [x] [SPIN](demo/text_recognition/spin) (AAAI 2021)

- [x] [RF-Learning](demo/text_recognition/rflearning) (ICDAR 2021)

***Text Spotting***

- [x] [Mask RCNN E2E](demo/text_spotting/mask_rcnn_spot/) 

- [x] [MANGO](demo/text_spotting/mango) (AAAI 2021)

***Information Extraction***

- [x] [TRIE](demo/text_ie/trie) (ACM MM 2020)

***Video Text Spotting***

- [x] [YORO](demo/videotext/yoro) (ACM MM 2019)

- FREE (to be released) (TIP 2021)

***Layout Recognition***

- VSR (to be released) (ICDAR 2021)

***Table Recognition***

- LGPMA (to be released) (ICDAR 2021)

## Development Environment
The recommended environment requirements can be found in [mmdetection](https://github.com/open-mmlab/mmdetection/). Follows are the lowest compatible environment.

| Basic Env   | version |
| :---------- | ------- |
| Python      | 3.6+    |
| cuda        | 10.0+   |
| cudnn       | 7.6.3+  |
| pytorch     | 1.3.0+  |
| torchvision | 0.4.1+  |
| opencv      | 3.0.0+  |

## Installation and Development Instruction 

To Download the repository and install the davarocr, please follow the instructions:

```shell
git clone https://github.com/hikopensource/DAVAR-Lab-OCR.git
cd DAVAR-Lab-OCR/
bash setup.sh
```

This script will automatically download and install the "mmdetection" and "mmcv-full". You can also manually install them followinging the [official instructions](https://github.com/open-mmlab/mmdetection/)

Going to the specific algorithm's directory to see more details.

## Problem solution and collection
For the problems existing in the process of installation and researching, we will reasonably collect them and provide corresponding solutions. Please refer to [FAQ.md](./docs/FAQ.md) for details. 


## Changelog

DavarOCR v0.1.0 was released in 23/07/2021.
Please refer to [Changelog.md](./docs/Changelog.md) for details and release history.


## License
This project is released under the [Apache 2.0 license](mmdetection/third_party/LICENSE)

## Copyright

The copyright of corresponding contributions of our implementations belongs to *Davar-Lab, Hikvision Research Institute, China*, and other codes from open source repository follows the original distributive licenses.

## Welcome to DAVAR-LAB!
See [latest news](https://davar-lab.github.io/) in DAVAR-Lab. If you have any question and suggestion, please feel free to contact us. Contact email: qiaoliang6@hikvision.com, xuyunlu@hikvision.com, chengzhanzhan@hikvision.com.
