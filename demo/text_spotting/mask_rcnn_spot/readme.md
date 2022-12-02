# Mask-RCNN Spotter
This code repository contains the implementation of a simple Mask-RCNN based Text Spotter. Many advanced text spotters are built based on such framework, e.g., 
- [Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes](https://arxiv.org/pdf/1908.08207.pdf) (ECCV 2018)
- [Towards Unconstrained End-to-End Text Spotting](https://arxiv.org/pdf/1908.09231.pdf) (ICCV 2019)
- [All You Need Is Boundary: Toward Arbitrary-Shaped Text Spotting](https://arxiv.org/pdf/1911.09550.pdf) (AAAI 2020)
- ...

## Preparing Dataset
Original images can be downloaded from: [Total-Text](https://github.com/cs-chan/Total-Text-Dataset "Total-Text") , [ICDAR2013](https://rrc.cvc.uab.es/?ch=2) , [ICDAR2015](https://rrc.cvc.uab.es/?ch=4), [ICDAR2017_MLT](https://rrc.cvc.uab.es/?ch=8).

The formatted training datalists can be found in [`demo/text_spotting/datalist`](../datalist)

## Train On Your Own Dataset
1.Download the pre-trained model, which was well trained on SynthText and COCO-Text.

2.Modify the paths (`ann_file`, `img_prefix`, `work_dir`, etc..) in the config files.

3.Modify the paths in training scripting and run the following bash command in the command line
``` shell
cd $DAVAR_LAB_OCR_ROOT$/demo/text_spotting/mask_rcnn_spot/
bash dist_train.sh
```
>Notice:We provide the implementation of online validation. If you want to close it to save training time, you may modify the startup script to add `--no-validate` command.

## Train From Scratch
If you want to re-implement the model's performance from scratch, please following these steps:

1.End-to-End pre-training using the SynthText and COCO-Text. See [`demo/text_spotting/mask_rcnn_spot/configs/mask_rcnn_r50_conv6_e2e_pretrain.py`](./configs/mask_rcnn_r50_conv6_e2e_pretrain.py) for more details.

2.Fine-tune model on the mixed real dataset (include:ICADR2013, ICDAR2015, ICDAR2017-MLT, Total-Text). See [`demo/text_spotting/mask_rcnn_spot/configs/mask_rcnn_r50_conv6_e2e_finetune_ic13.py`](./configs/mask_rcnn_r50_conv6_e2e_finetune_ic13.py) for more details.

>Notice:We provide the implementation of online validation, if you want to close it to save training time, you may modify the startup script to add `--no-validate` command.

## Offline Inference and Evaluation
We provide a demo of forward inference and evaluation. You can modify the parameter (`iou_constraint`, `lexicon_type`, etc..) in the testing script, and start testing:
``` shell
cd $DAVAR_LAB_OCR_ROOT$/demo/text_spotting/mask_rcnn_spot/tools/
bash test_ic13.sh
```

The offline evaluation tool can be found in [`davarocr/demo/text_spotting/evaluation/`](../evalution/).

## Visualization
We provide a script to visualize the intermediate output results of the model. You can modify the paths (`test_dataset`, `config_file`, etc..) in the script, and start generating visualization results:
``` shell
cd $DAVAR_LAB_OCR_ROOT$/demo/text_spotting/mask_rcnn_spot/tools/
python vis.py
```

Some visualization results are shown:

![./vis/img_225_text.jpg](./vis/img_225_text.jpg)
![./vis/img92_text.jpg](./vis/img92_text.jpg)

## Trained Model Download
All of the models are re-implemented and well trained in the based on the opensourced framework mmdetection.
>Note: The following trained model based on [mask_rcnn_r50_fpn+res32+bilstm+attention](./configs/mask_rcnn_r50_r32_e2e_pretrain.py) 
>uses only synthtext pre-training, and does not use random crop, color jitter, mix-train strategy, so the reported performance is slightly worse than that of [mask_rcnn_r50_fpn+conv6+bilstm+attention](./configs/mask_rcnn_r50_conv6_e2e_pretrain.py).

Results on various datasets and trained models download:
<table>
	<tr>
		<td>Pipeline</td>
		<td>Pretrained-Dataset</td>
		<td>Links</td>
	</tr>
	<tr>
		<td>mask_rcnn_r50_fpn+conv6+bilstm+attention</td>
		<td>SynthText<br>COCO-Text</td>
		<td><p><a href="./configs/mask_rcnn_r50_conv6_e2e_pretrain.py">cfg </a>, <a href="https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=jLte6Hd6">pth </a> (Access Code: yu09)</p></td>
	</tr>
	<tr>
		<td>mask_rcnn_r50_fpn+res32+bilstm+attention</td>
		<td>SynthText</td>
		<td><p><a href="./configs/mask_rcnn_r50_r32_e2e_pretrain.py">cfg </a>, <a href="https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=Icqq2yzo">pth </a> (Access Code: cAgC)</p></td>
	</tr>
</table>

<table>
	<tr>
		<td rowspan="2">Dataset</td>
		<td rowspan="2">Backbone</td>
		<td rowspan="2">Pretrained</td>
		<td rowspan="2">Finetune</td>
		<td rowspan="2">Test Scale</td>
		<td colspan="3">End-to-End</td>
		<td colspan="3">Word Spotting</td>
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
		<td>ResNet-50<br>Conv-6x</td>
		<td>SynthText<br>COCO-Text</td>
		<td>ICDAR2013<br>ICDAR2015<br>ICDAR2017_MLT<br>Total-Text</td>
		<td>L-1440</td>
		<td>82.1</td>
		<td>85.6</td>
		<td>86.1</td>
		<td>85.6</td>
		<td>89.9</td>
		<td>90.5</td>
		<td><p><a href="./configs/mask_rcnn_r50_conv6_e2e_finetune_ic13.py">cfg </a>, <a href="https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=y6J95GcV">pth </a> (Access Code: eFes)</p></td>
	</tr>
	<tr>
		<td>ICDAR2013</td>
		<td>ResNet-50<br>ResNet-32</td>
		<td>SynthText</td>
		<td>ICDAR2013<br>ICDAR2015<br>ICDAR2017_MLT<br>Total-Text</td>
		<td>L-1440</td>
		<td>82.7</td>
		<td>86.0</td>
		<td>86.6</td>
		<td>86.1</td>
		<td>90.4</td>
		<td>91.1</td>
		<td><p><a href="./configs/mask_rcnn_r50_r32_e2e_finetune_ic13.py">cfg </a>, <a href="https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=IY5EezaW">pth </a> (Access Code: xtD3)</p></td>
	</tr>
	<tr>
		<td>ICDAR2015</td>
		<td>ResNet-50<br>Conv-6x</td>
		<td>SynthText<br>COCO-Text</td>
		<td>ICDAR2013<br>ICDAR2015<br>ICDAR2017_MLT<br>Total-Text</td>
		<td>L-2000</td>
		<td>66.3</td>
		<td>75.3</td>
		<td>78.4</td>
		<td>66.7</td>
		<td>78.1</td>
		<td>81.7</td>
		<td><p><a href="./configs/mask_rcnn_r50_conv6_e2e_finetune_ic15.py">cfg </a>, <a href="https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=y6J95GcV">pth </a> (Access Code: eFes)</p></td>
	</tr>
	<tr>
		<td>ICDAR2015</td>
		<td>ResNet-50<br>ResNet-32</td>
		<td>SynthText</td>
		<td>ICDAR2013<br>ICDAR2015<br>ICDAR2017_MLT<br>Total-Text</td>
		<td>L-2000</td>
		<td>62.9</td>
		<td>72.2</td>
		<td>75.7</td>
		<td>63.5</td>
		<td>75.0</td>
		<td>79.1</td>
		<td><p><a href="./configs/mask_rcnn_r50_r32_e2e_finetune_ic15.py">cfg </a>, <a href="https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=KJXJj3TY">pth </a> (Access Code: 030W)</p></td>
	</tr>
</table>

<table>
	<tr>
		<td rowspan="2">Dataset</td>
		<td rowspan="2">Backbone</td>
		<td rowspan="2">Pretrained</td>
		<td rowspan="2">Finetune</td>
		<td rowspan="2">Test Scale</td>
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
		<td>ResNet-50<br>Conv-6x</td>
		<td>SynthText<br>COCO-Text</td>
		<td>ICDAR2013<br>ICDAR2015<br>ICDAR2017_MLT<br>Total-Text</td>
		<td>L-1350</td>
		<td>63.6</td>
		<td>72.2</td>
		<td>66.1</td>
		<td>76.5</td>
		<td><p><a href="./configs/mask_rcnn_r50_conv6_e2e_finetune_tt.py">cfg </a>, <a href="https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=y6J95GcV">pth </a> (Access Code: eFes)</p></td>
	</tr>
	<tr>
		<td>Total-Text</td>
		<td>ResNet-50<br>ResNet-32</td>
		<td>SynthText</td>
		<td>ICDAR2013<br>ICDAR2015<br>ICDAR2017_MLT<br>Total-Text</td>
		<td>L-1350</td>
		<td>62.8</td>
		<td>71.5</td>
		<td>65.2</td>
		<td>75.8</td>
		<td><p><a href="./configs/mask_rcnn_r50_r32_e2e_finetune_tt.py">cfg </a>, <a href="https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=umwkAXTx">pth </a> (Access Code: i3vP)</p></td>
	</tr>
</table>



## Citation:

``` markdown
@inproceedings{He_2017,
  title={Mask R-CNN},
  author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
  booktitle={2017 IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}
```

## License
This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright
If there is any suggestion and problem, please feel free to contact the author with qiaoliang6@hikvision.com or chengzhanzhan@hikvision.com.