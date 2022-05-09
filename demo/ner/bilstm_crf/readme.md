# BiLSTM+CRF

This code repository contains the implementations of the method [BiLSTM+CRF](http://arxiv.org/abs/1508.01991) (Arxiv 2015).

## Train From Scratch

If you want to train the model from scratch, please following these steps:

1.Firstly, because we adopt the vocab in transformers, so you need prepare the pretrained models that containd the vocab.txt:

-  [bert-base-cased](https://huggingface.co/bert-base-cased)

2.Secondly, modify the paths in model config (`demo/ner/bilstm_crf/bilstm_crf.py`), including the pretrained models paths, datasets paths, work space, etc. 

3.Thirdly, direct run `demo/ner/bilstm_crf/dist_train.sh`.

## Test

Given the trained model, direct run `demo/ner/bilstm_crf/test.sh` to test model.

## Trained Model Download

The results in paper used some tricks during training phase. So, the results might be slightly different from reported results.

Results on various datasets and trained models can be download as follows:

|   F1-score         | CoNLL2003| Resume     | Links       |
| :---------:        | :------: | :--------: | :---------: | 
| BiLSTM+CRF(paper)  |  84.3    |    -       | -   |
| BiLSTM+CRF         |  77.4    |   93.6     | [config](./configs/bilstm_crf.py), [pth](https://one.hikvision.com/#/link/PxkeemkIJvKThBJmKO4h) (Access Code：nfYT) |



## Citation

If you find this repository is helpful to your research, please feel free to cite us:

``` markdown
@article{DBLP:journals/corr/HuangXY15,
  author    = {Zhiheng Huang and
               Wei Xu and
               Kai Yu},
  title     = {Bidirectional {LSTM-CRF} Models for Sequence Tagging},
  journal   = {CoRR},
  volume    = {abs/1508.01991},
  year      = {2015},
}
```

## License

This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright

If there is any suggestion and problem, please feel free to contact the author with lipengfei27@hikvision.com or qiaoliang6@hikvision.com.