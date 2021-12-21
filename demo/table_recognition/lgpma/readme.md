# LGPMA

This code repository contains the implementations of the paper [LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask Alignment](https://arxiv.org/pdf/2105.06224.pdf) (ICDAR 2021).


## Preparing Dataset
Original images can be downloaded from: [pubtabnet](https://developer.ibm.com/exchanges/data/all/pubtabnet/).

The test datalist and the example of formatted training datalist can be found in `demo/table_recognition/datalist/`

The whole formatted training/validating datalist can be downloaded from [Hikvision One](https://one.hikvision.com/#/link/eqaugdX6YQgFKhUxF8U4) (Access Code: YF3O).

## Training
Modified the paths of "ann_file", "img_prefix", "pretrained_model" and "work_space" in the config files `demo/table_recognition/lgpma/config/lgpma_pub.py`.

Run the following bash command in the command line,
``` shell
cd $DAVAR_LAB_OCR_ROOT$/demo/table_recognition/lgpma/
bash dist_train.sh
```

## Offline Inference and Evaluation
We provide a demo of forward inference and evaluation on PubTabNet dataset. You can modify the paths (`savepath`, `config_file`, `checkpoint_file`) in test script, and start testing:

``` shell
python test_pub.py 
```

Some visualization of detection results are shown:

![./vis/PMC2871264_002_00.png](./vis/PMC2871264_002_00.png)

![./vis/PMC3160368_005_00.png](./vis/PMC3160368_005_00.png)

![./vis/PMC3250619_005_01.png](./vis/PMC3250619_005_01.png)

![./vis/PMC3551656_004_00.png](./vis/PMC3551656_004_00.png)

![./vis/PMC3568059_003_00.png](./vis/PMC3568059_003_00.png)

![./vis/PMC3824233_004_00.png](./vis/PMC3824233_004_00.png)

The offline evaluation tool can be found in `demo/table_recognition/lgpma/tools/eval_pub/`

## Trained Model Download
All of the models are re-implemented and well trained in the based on the opensourced framework mmdetection. So, the results might be slightly different from reported results.

Results on various datasets and trained models download:

| Dataset                | Test Scale     | TEDS-struc| Links |
|-----------------------|----------------|-----------|----------------------|
| PubTabNet(reported)    | L-768| 96.7      |                   |    |
| PubTabNet             | 1.5x    | 96.7      | [config](configs/lgpma_pub.py), [pth](https://one.hikvision.com/#/link/u9YgYyoPW3hLw6iolFoA) (Access Code: zUoX)| 

> The release model only contains structure-level result. You may use the [text recognition module](../../text_recognition) for the complete result.

> The Trained Model on dataset SciTSR and ICDAR 2013 will release soon.

## Citation

If you find this repository is helpful to your research, please feel free to cite us:

``` markdown
@inproceedings{qiao2021icdar21,
  title={LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask Alignment},
  author={Qiao, Liang and Li, Zaisheng and Cheng, Zhanzhan and Zhang, Peng and Pu, Shiliang and Niu, Yi and Ren, Wenqi and Tan, Wenming and Wu, Fei},
  booktitle={Document Analysis and Recognition-ICDAR 2021, 16th International Conference, Lausanne, Switzerland, September 5–10, 2021, Proceedings, Part I},
  pages={99-114},
  year={2021}
}
```
## License
This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright
If there is any suggestion and problem, please feel free to contact the author with qiaoliang6@hikvision.com, lizaisheng@hikvision.com or chengzhanzhan@hikvision.com.
