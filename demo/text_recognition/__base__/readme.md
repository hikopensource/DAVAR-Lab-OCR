# Recognition Base Model

## 1. Introduction

This code repository contains an implementation of (CRNN:[An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf))(TPAMI) and  (Res-Bilstm-Attn:[What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://arxiv.org/abs/1904.01906v4.pdf)) (ICCV 2019).

## 2. Preparing Dataset

### Train Dataset

| Dataset | Samples |                         Description                          |                       Release                       |
| :-----: | :-----: | :----------------------------------------------------------: | :-------------------------------------------------: |
| MJSynth | 8919257 |          Scene text recognition synthetic data set           | [Link](https://www.robots.ox.ac.uk/~vgg/data/text/) |
| SynText | 7266164 | A synthesized by scene text dataset, and the text is cropped from the large image |   [Link](https://github.com/ankush-me/SynthText)    |

### Validation Dataset
|  Testset  | Instance Number |   Note    |
| :-------: | :-------------: | :-------: |
|  IIIT5K   |      3000       |  regular  |
|    SVT    |       647       |  regular  |
| IC03_860  |       860       |  regular  |
| IC13_857  |       857       |  regular  |
| IC15_1811 |      1811       | irregular |
|   SVTP    |       645       | irregular |
|  CUTE80   |       288       | irregular |

### Test Dataset

|  Testset  | Instance Number |   Note    |
| :-------: | :-------------: | :-------: |
|  IIIT5K   |      3000       |  regular  |
|    SVT    |       647       |  regular  |
| IC03_860  |       860       |  regular  |
| IC13_857  |       857       |  regular  |
| IC15_1811 |      1811       | irregular |
|   SVTP    |       645       | irregular |
|  CUTE80   |       288       | irregular |



## 3. Getting Started
### Preparation
A quick start is to use above lmdb-formatted datasets that contain the full benchmarks for scene text recognition tasks as belows.
```
Data Type: LMDB

File storage format:
   |-- train           
   |   |-- MJ
   |   |-- ST
   |-- validation
   |   |-- mixture
   |-- evaluation
   |   |-- mixture
```



### Training 

Run the following bash command in the command line,

```
cd .
bash ./train_script/train_att.sh 

cd .
bash ./train_script/train_crnn.sh 
```
> We provide the implementation of online validation. If you want to close it to save training time, you may modify the startup script to add `--no-validate` command.

### Evaluation

```
cd .
bash ./train.sh
```



## 4. Results

### Evaluation

<table>
    <tr>
        <td><strong><center>Methods</center></strong></td>
        <td colspan="4"><strong><center>Regular Text</center></strong></td>
        <td colspan="3"><strong><center>Irregular Text</center></strong></td> 
        <td colspan="2"><center><strong>Download</center></strong></td>
    <tr>
	<tr>
        <td><center> Name </center></td>
        <td><center> IIIT5K </center></td>
        <td><center> SVT </center></td>
        <td><center> IC03 </center></td>
        <td><center> IC13 </center></td>
        <td><center> IC15 </center></td>
        <td><center> SVTP </center></td>
        <td><center>CUTE80</center></td>
        <td><center>Config</center></td>
        <td><center>Model</center></td>
	<tr>
    <tr>
		<td><center> CRNN(Report) </center></td>
        <td><center> 86.2 </center></td>
        <td><center> 86.0 </center></td>
        <td><center> 94.4 </center></td>
        <td><center> 92.6 </center></td>
        <td><center> 73.6 </center></td>
        <td><center> 76.0 </center></td>
        <td><center> 72.2 </center></td>
        <td><center><p>-</p></center></td>
        <td><center><p>-</p></center></td>
        </center></td>
	<tr>
    <tr>
		<td><center> CRNN </center></td>
        <td><center> 93.3 </center></td>
        <td><center> 87.5 </center></td>
        <td><center> 92.6 </center></td>
        <td><center> 92.4 </center></td>
        <td><center> 78.1 </center></td>
        <td><center> 78.9 </center></td>
        <td><center> 80.6 </center></td>
        <td><center><p><a href="./res32_bilstm_ctc.py"> Config </a></p></center>
        <td><center><p> pth <a href="https://pan.baidu.com/s/1Ad91GOxnFm3XcGZJoR8XAQ">BaiduYunPan </a>(Code:yc98), <a href="https://drive.google.com/drive/folders/1VNYKzqSQ1tNPfBvglmfjDfaNWR6RQSFj?usp=sharing">Google Drive </a></p></center></td>
	<tr>
    <tr>
		<td><center> Attention(Report) </center></td>
        <td><center> 86.6 </center></td>
        <td><center> 86.2 </center></td>
        <td><center> 94.1 </center></td>
        <td><center> 92.8 </center></td>
        <td><center> 75.6 </center></td>
        <td><center> 76.4 </center></td>
        <td><center> 72.6 </center></td>
        <td><center><p>-</p></center></td>
        <td><center><p>-</p></center></td>
        </center></td>
	<tr>
    <tr>
		<td><center> Attention </center></td>
        <td><center> 94.5 </center></td>
        <td><center> 89.0 </center></td>
        <td><center> 94.5 </center></td>
        <td><center> 94.1 </center></td>
        <td><center> 81.7 </center></td>
        <td><center> 82.5 </center></td>
        <td><center> 81.9 </center></td>
        <td><center><p><a href="./res32_bilstm_attn.py"> Config </a></p></center>
        <td><center><p> pth <a href="https://pan.baidu.com/s/1B5pBYYXTgdXNQNfue_ET9A">BaiduYunPan </a> (Code:mdtf),  <a href="https://drive.google.com/drive/folders/1ZL_wyWq2AFURRN5Bcj6EN85-71b7MXVr?usp=sharing"> Google Drive</a> </p></center></td>
	<tr>
<table>


### Visualization
Here is the picture for result visualization. 

<img src="./figs/visualization.png" width = "80%" alt="visualization"/>

## Citation

``` markdown
@article{CRNN,
  author={Baoguang Shi and Xiang Bai and Cong Yao},
  title={An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition},
  journal={TPAMI},
  volume={39},
  number={11},
  pages={2298--2304},
  year={2017},
}

@inproceedings{Wrong,
  author={Jeonghun Baek and Geewook Kim and Junyeop Lee and Sungrae Park and Dongyoon Han and Sangdoo Yun and Seong Joon Oh and Hwalsuk Lee},
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  booktitle={ICCV 2019},
  pages={4714--4722},
  publisher={{IEEE}},
  year={2019},
}
```

## License

This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright

If there is any suggestion and problem, please feel free to contact the author with jianghui11@hikvision.com or chengzhanzhan@hikvision.com.