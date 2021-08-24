# Text Perceptron Detector

This code repository contains the implementations of the paper [Text Perceptron: Towards End-to-End Arbitrary-Shaped Text Spotting](https://arxiv.org/pdf/2002.06820.pdf) (AAAI 2020).

> To date, the detection part in the Text Perceptron is re-implemented based on the common repository [Davar-OCR](https://github.com/hikopensource/DAVAR-Lab-OCR), and the recognition part will be released in the future.


## Preparing Dataset
Original images can be downloaded from: [Total-Text](https://github.com/cs-chan/Total-Text-Dataset "Total-Text") , [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector).

The formatted training datalist and test datalist can be found in `demo/text_detection/datalist/`

## Training
Modified the paths of "imgs"/ "pretrained_model"/ "work_space" in the config files `demo/text_detection/text_perceptron_det/config/tp_r50_3stages_enlarge.py`.

Run the following bash command in the command line,
``` bash
>>> cd $DAVAR_LAB_OCR_ROOT$/demo/text_detection/text_perceptron_det/
>>> bash dist_train.sh
```

> We provide the implementation of online validation. If you want to close it to save training time, you may modify the startup script to add `--no-validate` command.

## Offline Inference and Evaluation
We provide a demo of forward inference and visualization. You can modify the paths (`test_dataset`, `image_prefix`, etc.) in the testing script, and start testing:
``` bash
>>> python test.py 
```
Some visualization of detection results are shown:

![./vis/pred_img499.jpg](./vis/pred_img499.jpg)
![./vis/pred_img566.jpg](./vis/pred_img566.jpg)

### Offline Evaluation

The offline evaluation tool can be found in [`davarocr/demo/text_detection/evaluation/`](../evalution/).

## Trained Model Download
All of the models are re-implemented and well trained in the based on the opensourced framework mmdetection. So, the results might be slightly different from reported results.

Results on various datasets and trained models download:

|   Dataset                          | Backbone                 | Pretrained | Test Scale| Precision | Recall | Hmean | Links               |
| -----------------------------------|--------------------------| ---------- |  --------- | --------- | ------ | ----- | ------------------- |
| Total-Text (Reported)              | ResNet-50-3stages-enlarge| SynthText  |  L-1350 | 88.1      | 78.9   | 83.3  | -                   |
| Total-Text                         | ResNet-50-3stages-enlarge| SynthText  |  L-1350 | 89.0      | 81.1   | 84.8  | [config](config/tp_det_r50_3stages_enlarge_tt.py), [pth](https://pan.baidu.com/s/1lEvCgxc-0nEXIdE9GkKcWw ) (Access Code: 2345)|
| Total-Text                         | ResNet-50                | SynthText+IC17, [pth](https://pan.baidu.com/s/17lnY0shAtvDlHZXz_E1vSQ) (Access Code: 8dn0) | L-1350 |  84.5 | 80.4 | 82.4| [config](config/tp_det_r50.py), [pth](https://pan.baidu.com/s/1HhjysDTI7gMOqDDGk3sbJQ) (Access Code: p692)|
| SCUT-CTW1500 (Reported)            | ResNet-50-3stages-enlarge| SynthText  |  L-1250 | 88.7      | 78.2   | 83.1  | -                   |
| SCUT-CTW1500                       | ResNet-50-3stages-enlarge| SynthText  |   L-1250 | 86.2      | 79.9   | 82.9  | [config](config/tp_det_r50_3stages_enlarge_ctw.py), [pth](https://pan.baidu.com/s/1HfYLzuybdqDTChbPYuCgrg ) (Access Code: t2z9)|
| SCUT-CTW1500  (considers NOT CARE) | ResNet-50-3stages-enlarge| SynthText  |   L-1250 | 85.9      | 83.6   | 84.7  | [config](config/tp_det_r50_3stages_enlarge_ctw.py), [pth](https://pan.baidu.com/s/1HfYLzuybdqDTChbPYuCgrg ) (Access Code: t2z9)|
| SCUT-CTW1500                       | ResNet-50                | SynthText+IC17, [pth](https://pan.baidu.com/s/17lnY0shAtvDlHZXz_E1vSQ) (Access Code: 8dn0)|  L-1250 |  87.7 | 79.8 | 83.6| [config](config/tp_det_r50.py), [pth](https://pan.baidu.com/s/1dZ2Pa-I0JE3bNPUDGL70wA)(Access Code: duuo)|
| SCUT-CTW1500  (considers NOT CARE) | ResNet-50                | SynthText+IC17, [pth](https://pan.baidu.com/s/17lnY0shAtvDlHZXz_E1vSQ) (Access Code: 8dn0)|   L-1250 | 87.5 | 84.5 | 85.9| [config](config/tp_det_r50.py), [pth](https://pan.baidu.com/s/1dZ2Pa-I0JE3bNPUDGL70wA)(Access Code: duuo)|
> The original SCUT-CTW1500 was released to only support text detection task, where all text instances are considered. When this dataset was used in text spotting task, some instances (blurred or in other languages) were labeled as not care.   

> Note: Models are stored in BaiduYunPan, and can also be downloaded from [Google Drive](https://drive.google.com/drive/folders/1BuIt7fhzCh3kFXEVbtR1MS-kYVXmjsNq?usp=sharing)

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
This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright
If there is any suggestion and problem, please feel free to contact the author with qiaoliang6@hikvision.com or chengzhanzhan@hikvision.com.
