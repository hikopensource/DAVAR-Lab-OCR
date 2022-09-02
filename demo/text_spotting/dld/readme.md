# Dynamic Low-Resolution Distillation
This code repository contains the implementations of the paper [Dynamic Low-Resolution Distillation for Cost-Efficient End-to-End Text Spotting](https://arxiv.org/pdf/2207.06694.pdf) (ECCV 2022).

## Preparing Dataset
Original images can be downloaded from: [Total-Text](https://github.com/cs-chan/Total-Text-Dataset "Total-Text") , [ICDAR2013](https://rrc.cvc.uab.es/?ch=2) , [ICDAR2015](https://rrc.cvc.uab.es/?ch=4), [ICDAR2017_MLT](https://rrc.cvc.uab.es/?ch=8).

The formatted training datalists can be found in [`demo/text_spotting/datalist`](../datalist)

## Train From Scratch
If you want to re-implement the model's performance from scratch, please following these steps:

1.Download the pre-trained model, which was well trained on SynthText & COCO-Text ([pth](https://one.hikvision.com/#/link/asT46Ufzfbf7QTvotanK) (Access Code: ngPI)). See [`demo/text_spotting/mask_rcnn_spot/readme.md`](../mask_rcnn_spot/readme.md) for more details.

2.Train the multi-scale teacher model using the ICDAR2013, ICDAR2015, ICDAR2017-MLT and Total-Text based on the pre-trained model in step-1 (L307 in `mask_rcnn_pretrain_teacher.py`). The teacher model is also used as the Vanilla Multi-Scale competitors.  See [`demo/text_spotting/dld/configs/mask_rcnn_pretrain_teacher.py`](./configs/mask_rcnn_pretrain_teacher.py) for more details.

Just modify the required path in the config file (`img_prefixes`, `ann_files`, `work_dir`, `load_from`, etc.) and then run the following script:
``` shell
cd $DAVAR_LAB_OCR_ROOT$/demo/text_spotting/dld/
bash dist_train_teacher.sh
```

3.Initialize teacher and student models with the trained models obtained in step-2 (L360-361 in `mask_rcnn_distill.py`), and then end-to-end distill student model on the mixed real dataset (include:ICADR2013, ICDAR2015, ICDAR2017-MLT and Total-Text).  The results on separate testing dataset are reported based on the same model.   See [`demo/text_spotting/dld/configs/mask_rcnn_distill.py`](./configs/mask_rcnn_distill.py) for more details.

Just modify the required path in the config file (`img_prefixes`, `ann_files`, `work_dir`, `load_from`, etc.) and then run the following script:
``` shell
cd $DAVAR_LAB_OCR_ROOT$/demo/text_spotting/dld/
bash dist_train_distill.sh
```

>Notice:We provide the implementation of online validation, if you want to close it to save training time, you may modify the startup script to add `--no-validate` command.

## Offline Inference and Evaluation
We provide a demo of forward inference and evaluation. You can modify the parameter (`iou_constraint`, `lexicon_type`, etc..) in the testing script, and start testing. For example:

``` shell
cd $DAVAR_LAB_OCR_ROOT$/demo/text_spotting/mask_rcnn_spot/tools/
bash test_ic13.sh
```

The offline evaluation tool can be found in [`davarocr/demo/text_spotting/evaluation/`](../evalution/).

## Trained Model Download
All of the models are re-implemented and well trained in the based on the opensourced framework mmdetection.

Results on various datasets and trained models download:

<table>
	<tr>
		<td rowspan="2">Dataset</td>
		<td rowspan="2">Training Method</td>
		<td rowspan="2">Input Size</td>
		<td colspan="3">End-to-End</td>
		<td colspan="3">Word Spotting</td>
		<td rowspan="2">FLOPS</td>
		<td rowspan="2">Links</td>
	</tr>
	<tr>
		<td>General</td>
		<td>Weak</td>
		<td>Strong</td>
		<td>General</td>
		<td>Weak</td>
		<td>Strong</td>
	</tr>
	<tr>
		<td>ICDAR2013</td>
		<td>Vanilla Multi-Scale</td>
		<td>S-768</td>
		<td>82.9</td>
		<td>86.6</td>
		<td>86.9</td>
		<td>86.3</td>
		<td>91.0</td>
		<td>91.4</td>
		<td>142.9G</td>
		<td><p><a href="./configs/mask_rcnn_pretrain_teacher.py">cfg </a>, <a href="https://one.hikvision.com/#/link/rErY9rkFYldN2MNYn2fp">pth </a> (Access Code: 7OXC)</p></td>
	</tr>
	<tr>
		<td>ICDAR2013</td>
		<td>DLD (γ=0.1)</td>
		<td>Dynamic</td>
		<td>82.7</td>
		<td>85.7</td>
		<td>86.5</td>
		<td>86.1</td>
		<td>89.9</td>
		<td>90.9</td>
		<td>71.5G</td>
		<td><p><a href="./configs/mask_rcnn_distill.py">cfg </a>, <a href="https://one.hikvision.com/#/link/lknYDorAPPQpKEwdsTvn">pth </a> (Access Code: EhvH)</p></td>
	</tr>
	<tr>
		<td>ICDAR2013</td>
		<td>DLD (γ=0.3)</td>
		<td>Dynamic</td>
		<td>81.6</td>
		<td>84.4</td>
		<td>85.6</td>
		<td>84.9</td>
		<td>88.6</td>
		<td>90.0</td>
		<td>41.6G</td>
		<td><p><a href="./configs/mask_rcnn_distill.py">cfg </a>, <a href="https://one.hikvision.com/#/link/rPn2acvl2PdC9PilNr8w">pth </a> (Access Code: NNOZ)</p></td>
	</tr>
	<tr>
		<td>ICDAR2015</td>
		<td>Vanilla Multi-Scale</td>
		<td>S-1280</td>
		<td>69.5</td>
		<td>74.4</td>
		<td>78.0</td>
		<td>71.7</td>
		<td>77.2</td>
		<td>81.4</td>
		<td>517.2G</td>
		<td><p><a href="./configs/mask_rcnn_pretrain_teacher.py">cfg </a>, <a href="https://one.hikvision.com/#/link/rErY9rkFYldN2MNYn2fp">pth </a> (Access Code: 7OXC)</p></td>
	</tr>
	<tr>
		<td>ICDAR2015</td>
		<td>DLD (γ=0.1)</td>
		<td>Dynamic</td>
		<td>70.9</td>
		<td>75.7</td>
		<td>79.0</td>
		<td>73.3</td>
		<td>78.6</td>
		<td>82.4</td>
		<td>298.8G</td>
		<td><p><a href="./configs/mask_rcnn_distill.py">cfg </a>, <a href="https://one.hikvision.com/#/link/lknYDorAPPQpKEwdsTvn">pth </a> (Access Code: EhvH)</p></td>
	</tr>
	<tr>
		<td>ICDAR2015</td>
		<td>DLD (γ=0.3)</td>
		<td>Dynamic</td>
		<td>69.3</td>
		<td>73.5</td>
		<td>78.1</td>
		<td>71.2</td>
		<td>76.4</td>
		<td>81.1</td>
		<td>148.3G</td>
		<td><p><a href="./configs/mask_rcnn_distill.py">cfg </a>, <a href="https://one.hikvision.com/#/link/rPn2acvl2PdC9PilNr8w">pth </a> (Access Code: NNOZ)</p></td>
	</tr>
</table>

<table>
	<tr>
		<td rowspan="2">Dataset</td>
		<td rowspan="2">Training Method</td>
		<td rowspan="2">Input Size</td>
		<td colspan="2">End-to-End</td>
		<td colspan="2">Word Spotting</td>
		<td rowspan="2">Links</td>
	</tr>
	<tr>
		<td>None</td>
		<td>Full</td>
		<td>None</td>
		<td>Full</td>
	</tr>
	<tr>
		<td>Total-Text</td>
		<td>Vanilla Multi-Scale</td>
		<td>S-896</td>
		<td>62.3</td>
		<td>71.4</td>
		<td>65.2</td>
		<td>75.9</td>
		<td>206.7G</td>
		<td><p><a href="./configs/mask_rcnn_pretrain_teacher.py">cfg </a>, <a href="https://one.hikvision.com/#/link/rErY9rkFYldN2MNYn2fp">pth </a> (Access Code: 7OXC)</p></td>
	</tr>
	<tr>
		<td>Total-Text</td>
		<td>DLD (γ=0.1)</td>
		<td>Dynamic</td>
		<td>63.9</td>
		<td>73.7</td>
		<td>66.4</td>
		<td>77.8</td>
		<td>103.0G</td>
		<td><p><a href="./configs/mask_rcnn_distill.py">cfg </a>, <a href="https://one.hikvision.com/#/link/lknYDorAPPQpKEwdsTvn">pth </a> (Access Code: EhvH)</p></td>
	</tr>
	<tr>
		<td>Total-Text</td>
		<td>DLD (γ=0.3)</td>
		<td>Dynamic</td>
		<td>61.9</td>
		<td>71.9</td>
		<td>64.0</td>
		<td>75.9</td>
		<td>62.1G</td>
		<td><p><a href="./configs/mask_rcnn_distill.py">cfg </a>, <a href="https://one.hikvision.com/#/link/rPn2acvl2PdC9PilNr8w">pth </a> (Access Code: NNOZ)</p></td>
	</tr>
</table>

## Citation:

``` markdown
@inproceedings{chen2022dynamic,
  title={Dynamic Low-Resolution Distillation for Cost-Efficient End-to-End Text Spotting},
  author={Chen, Ying and Qiao, Liang and Cheng, Zhanzhan and Pu, Shiliang and Niu, Yi and Li, Xi},
  booktitle={ECCV},
  year={2022}
}
```

## License
This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright
If there is any suggestion and problem, please feel free to contact the author with qiaoliang6@hikvision.com or chengzhanzhan@hikvision.com.