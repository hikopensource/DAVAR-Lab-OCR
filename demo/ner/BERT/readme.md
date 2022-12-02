# BERT+CRF

This code repository contains the implementations of the method BERT+Softmax/Span/CRF. The Original BERT (NAACL-HLT, 2019) paper can be found in [link](https://doi.org/10.18653/v1/n19-1423)


## Train From Scratch

If you want to train the model from scratch, please following these steps:

1.Firstly, prepare the pretrained models:

-  [bert-base-cased](https://huggingface.co/bert-base-cased)
-  [bert-base-chinese](https://huggingface.co/bert-base-chinese)

2.Secondly, modify the paths in model config (`demo/ner/BERT/bert_softmax(span/crf).py`), including the pretrained models paths, datasets paths, work space, etc. 

3.Thirdly, direct run `demo/ner/BERT/dist_train.sh`.

## Test

Given the trained model, direct run `demo/ner/BERT/test.sh` to test model.

## Trained Model Download

All of the models are re-implemented and well trained based on the opensourced framework mmdetection. So, the results might be slightly different from reported results.

Results on various datasets and trained models can be download as follows:

|   F1-score         | CoNLL2003| Resume     | Links       |
| :---------:        | :------: | :--------: | :---------: | 
| BERT+Softmax(paper)|  92.4    |    -       | -   |
| BERT+Softmax       |  92.1    |   96.4     | [config](./configs/bert_softmax.py), [pth](https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=hDCGBXwH&) (Access Code：agh1) |
| BERT+Span          |  92.1    |   96.2     |  [config](./configs/bert_span.py), [pth](https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=FY2dFLtw) (Access Code：308S)  |
| BERT+CRF           |  92.6    |   96.7     |  [config](./configs/bert_crf.py), [pth](https://drive.hikvision.com/hcs/controller/hik-manage/fileDownload?link=zlUi4wcO) (Access Code：1075)  |


## Citation


``` markdown
@inproceedings{devlin2019bert,
	 author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  booktitle = {NAACL-HLT},
  pages     = {4171--4186},
  year      = {2019},
}
```

## License

This project is released under the [Apache 2.0 license](../../../davar_ocr/LICENSE)

## Copyright

If there is any suggestion and problem, please feel free to contact the author with lipengfei27@hikvision.com or qiaoliang6@hikvision.com.