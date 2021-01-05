# DAVAR-OCR

This the OCR repository of [DAVAR Lab](https://davar-lab.github.io/), from Hikvision Research Institute, China. 

We begin to maintain this code repository to release the official implementations of our recent academic publishments in OCR.

> Note: Due to the policy limits of the company. All of the codes were re-implemented based on the open-source frameworks, [mmdetection-1.2.0](https://github.com/open-mmlab/mmdetection/releases/tag/v1.2.0) and [mmcv-0.4.2](https://github.com/open-mmlab/mmcv/releases/tag/v0.4.2), from [open-mmlab](https://github.com/open-mmlab "open-mmlab"). Therefore, the demonstrated results might be slightly different from the reported performances.

## Implementations
To date, we have released / will release the following algorithms:

- MANGO (to be released) (AAAI 2021)

- SPIN (to be released) (AAAI 2021)

- FREE (to be released) (TIP 2020)

- TRIE (to be released) (ACM MM 2020)

- [Text Perceptron](mmdetection/third_party/text_perceptron) (AAAI 2020)

- YORO (to be released) (ACM MM 2019)

## Development Environment

| Basic Env   | version |
| :---------- | ------- |
| Python      | 3.6     |
| cuda        | 10.0    |
| cudnn       | 7.6.3   |
| opencv      | 3.4.9   |
| pytorch     | 1.3.0   |
| torchvision | 0.4.1   |

## Installation and Development Instruction 
We keep the main part of mmdetection and mmcv exactly same with the official version. Each algorithm is stored under `mmdetection/third_party` in a separate directory structure.

To Download the repository and install the mmdetection and mmcv, please follow the instructions:
``` basic
>>> git clone https://github.com/hikopensource/DAVAR-Lab-OCR.git
>>> cd DAVAR-Lab-OCR/
>>> bash setup.sh
```
If you want to run some model, you can uncomment the corresponding importing statement in `mmdetection/third_party/__init__.py` directly in develop mode.

For example, if you want to use the model of Text Perceptron, you could uncomment the line of `from .text_perceptron import *` in `mmdetection/third_party/__init__.py`.

Going to the specifc algorithm's directory to see more details.
## License
This project is released under the [Apache 2.0 license](mmdetection/third_party/LICENSE)

## Copyright

The copyright of corresponding contributions of our third-party modules belongs to *Davar-Lab, Hikvision Research Institute, China*, and other codes from open source repository follows the original distributive licenses.

## Welcome to DAVAR-LAB!
See [latest news](https://davar-lab.github.io/) in DAVAR-Lab.
