"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    ner_loader.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import os
import json
import transformers
from davarocr.davar_nlp_common.datasets import BaseLoader
from davarocr.davar_nlp_common.datasets.builder import LOADERS


@LOADERS.register_module()
class NERLoader(BaseLoader):
    """ Load annotation from annotation file.
    """
    def __init__(self,
                 max_len,
                 truncation,
                 stride,
                 tokenizer=None,
                 with_multi_labels=False,
                 token_labels_level=0,
                 keys=[],
                 with_label=True,
                 **kwargs):
        """
        Args:
            max_len(int): the text's max length
            truncation(bool): whether truncate on text.
            stride(int): sliding stride size.
            with_multi_labels(bool): whether load multi token labels
            token_labels_level(int): load token_labels'level.
            keys (list): the maintain key list in final result.
            with_label (bool): whether load labels.
        """
        self.max_len = max_len
        self.truncation = truncation
        self.stride = stride
        self.with_multi_labels = with_multi_labels
        self.token_labels_level = token_labels_level
        self.keys = keys
        self.with_label = with_label
        self.tokenizer = tokenizer
        if tokenizer is not None:
            assert isinstance(tokenizer, dict)
            assert 'model_name_or_path' in tokenizer
            tokenizer_type = tokenizer.get('type','AutoTokenizer')
            if 'type' in tokenizer:
                tokenizer.pop('type')
            model_name_or_path = tokenizer['model_name_or_path']
            tokenizer.pop('model_name_or_path')
            self.tokenizer = getattr(transformers, tokenizer_type).from_pretrained(model_name_or_path,
                                                    **tokenizer)

        assert isinstance(self.max_len, int)
        assert isinstance(self.truncation, bool)
        assert isinstance(self.stride, int)
        assert self.stride > 0, self.stride
        assert isinstance(with_multi_labels, bool)
        assert isinstance(token_labels_level, int)
        super().__init__(**kwargs)

    def _load(self, ann_file):
        lines = []
        if os.path.isdir(ann_file):#is dir
            ann_file_list = os.listdir(ann_file)
        elif isinstance(ann_file, (list, tuple)):#
            ann_file_list = ann_file
        else:#single file
            ann_file_list = [ann_file]
        for ann in ann_file_list:
            json_file = json.loads(open(ann, encoding='utf-8').read())
            for key, value in json_file.items():
                nlp_ann = value['nlp_ann']
                texts = nlp_ann['tokens'][0]
                if not texts:
                    continue
                if self.with_label:
                    if 'tokens_labels' in nlp_ann:
                        token_labels = nlp_ann['tokens_labels'][0]
                        assert len(texts) == len(token_labels)

                        if not self.with_multi_labels:#get single level tokens using token_labels_level
                            if isinstance(token_labels[0], list):
                                assert self.token_labels_level < len(token_labels[0])
                                token_labels = [label[self.token_labels_level] for label in nlp_ann['tokens_labels'][0]]
                    else:
                        token_labels = nlp_ann['entities'][0]
                
                tokens_length = []
                for i, word in enumerate(texts):
                    if self.tokenizer is not None:
                        token = self.tokenizer.tokenize(word)
                        tokens_length.append(len(token))
                    else:
                        tokens_length.append(1)
                tokens_length.append(0)
                if self.truncation:
                    #truncation
                    length = 0
                    truncation_texts = []
                    truncation_index = 0
                    for i, word in enumerate(texts):
                        if length + tokens_length[i] > self.max_len:
                            truncation_index = i
                            break
                        truncation_texts.append(word)
                        length += tokens_length[i]
                        
                    #texts = texts[:truncation_index]
                    if 'tokens_labels' in nlp_ann:
                        token_labels = token_labels[:truncation_index]
                    else:
                        label = []
                        for token_label in token_labels:
                            if len(token_label)==3:
                                if token_label[2]<=truncation_index:
                                    label.append([token_label[0],token_label[1],token_label[2]])
                            else:
                                if token_label[1][-1]<=truncation_index:
                                    label.append(token_label)
                        token_labels = label
                    text_range = (0, truncation_index)
                    if self.with_label:
                        line = {"tokens":truncation_texts,"token_labels":token_labels,\
                                    "id":key,"range":text_range}
                    else:
                        line = {"tokens":truncation_texts,\
                                    "id":key,"range":text_range}

                    if self.keys:
                        for key in self.keys:
                            key_value = nlp_ann[key][0]
                            line.update({key:key_value[:truncation_texts]})
                    lines.append(line)
                else:
                    #sliding window
                    if self.tokenizer is None:
                        for i in range(0, len(texts), self.stride):
                            if i + self.max_len > len(texts):
                                start = i
                                end = len(texts)
                            else:
                                start = i
                                end = i + self.max_len
                            text = texts[start:end]
                            text_range = (start, end)
                            if self.with_label:
                                if 'tokens_labels' in nlp_ann:
                                    label = token_labels[start:end]
                                else:
                                    label = []
                                    for token_label in token_labels:
                                        if len(token_label) == 3:
                                            if token_label[1] >= start and token_label[2]<=end:
                                                label.append([token_label[0],token_label[1]-start,token_label[2]-start])
                                        else:
                                            if token_label[1][0] >= start and token_label[1][-1]<=end:
                                                label.append([token_label[0],list(map(lambda x:x-start,token_label[1]))])

                                line = {"tokens":text,"token_labels":label,\
                                            "id":key,"range":text_range}
                            else:
                                line = {"tokens":text,\
                                            "id":key,"range":text_range}
                            if self.keys:
                                for _key in self.keys:
                                    key_value = nlp_ann[_key][0]
                                    line.update({_key:key_value[start:end]})
                            lines.append(line)
                    else:
                        length = 0
                        sliding_texts = []
                        truncation_texts = []
                        text_ranges = []
                        start = 0
                        for i, word in enumerate(texts):
                            if length + tokens_length[i] > self.max_len:
                                text_ranges.append((start, i))
                                assert length == sum(tokens_length[start:i]), (length)
                                sliding_texts.append(truncation_texts)
                                truncation_texts = []
                                length = 0
                                start = i
                            truncation_texts.append(word)
                            length += tokens_length[i]
                        if length > 0:
                            sliding_texts.append(truncation_texts)
                            text_ranges.append((start,len(texts)))
                        for text, text_range in zip(sliding_texts, text_ranges):
                            start, end = text_range
                            if self.with_label:
                                if 'tokens_labels' in nlp_ann:
                                    label = token_labels[start:end]
                                else:
                                    label = []
                                    for token_label in token_labels:
                                        if len(token_label) == 3:
                                            if token_label[1] >= start and token_label[2]<=end:
                                                label.append([token_label[0],token_label[1]-start,token_label[2]-start])
                                        else:
                                            if token_label[1][0] >= start and token_label[1][-1]<=end:
                                                label.append([token_label[0],list(map(lambda x:x-start,token_label[1]))])
                                line = {"tokens":text,"token_labels":label,\
                                            "id":key,"range":text_range}
                            else:
                                line = {"tokens":text,\
                                            "id":key,"range":text_range}
                            if self.keys:
                                for _key in self.keys:
                                    key_value = nlp_ann[_key][0]
                                    line.update({_key:key_value[start:end]})
                            lines.append(line)

        return lines
