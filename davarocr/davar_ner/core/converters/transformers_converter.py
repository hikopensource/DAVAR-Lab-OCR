"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    transformers_converter.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import json
import transformers
from transformers import AutoTokenizer
from seqeval.scheme import Tokens, IOBES
import numpy as np
from davarocr.davar_common.core import CONVERTERS
from .base_converter import BaseConverter


@CONVERTERS.register_module()
class TransformersConverter(BaseConverter):
    """ Transformers converter
    """
    def __init__(self,
                 model_name_or_path,
                 label_list,
                 do_lower_case=True,
                 annotation_type='bioes',
                 add_start_end=True,
                 max_len=None,
                 use_auto=True,
                 tokenizer='BertTokenizer',
                 use_custom_tokenize=False,
                 only_label_first_subword=False,
                 pad_token_label=0,
                 never_split=None,
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 unk_token="[UNK]",
                 pad_token="[PAD]",
                 cls_token_segment_id = 1,
                 **kwargs):
        """
        Args:
            model_name_or_path (str): tokenizer's path.
            label_list (list[str]): label list, used for convertion between label and id .
            do_lower_case (bool): tokenizer's param, whether convert string to lowercase.
            annotation_type (str): BIOES(B-begin, I-inside, O-outside, E-end, S-single)
            add_start_end (bool): whether add start and end label id.
            max_len (int): the text's max length, truncate when text length exceeds this value.
            use_auto (bool): whether use AutoTokenizer in transformers.
            tokenizer (str): tokenizer's name in transformers, this will be used when use_auto is set to false.
            use_custom_tokenize (bool): whether use custom tokenize function.
            only_label_first_subword (bool): when text tokenize to subword, only first subword will have label.
            pad_token_label (int): label corresponding to the pad_token.
            never_split (None or list): the token in never_split will pass in tokenize process.
            cls_token (str): the cls token.
            sep_token (str): the sep token.
            unk_token (str): the unk token.
            pad_token (str): the pad token.
            cls_token_segment_id (str): token_type id corresponding to the cls token.
        """
        super().__init__(**kwargs)
        if use_auto:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                    do_lower_case=do_lower_case,
                                                    never_split=never_split)#load tokenizer
        else:
            self.tokenizer = getattr(transformers, tokenizer).from_pretrained(model_name_or_path,
                                                    do_lower_case=do_lower_case,
                                                    never_split=never_split)#load tokenizer
        self.use_custom_tokenize = use_custom_tokenize
        self.never_split = never_split
        self.label_list = label_list
        self.annotation_type = annotation_type

        self.cls_token_at_end = False
        self.cls_token = cls_token
        self.cls_token_segment_id = cls_token_segment_id
        self.sep_token = sep_token
        self.pad_on_left = False
        self.pad_token = pad_token
        self.pad_token_segment_id = 0
        self.sequence_a_segment_id = 0
        self.mask_padding_with_zero = True
        self.max_len = max_len
        self.add_start_end = add_start_end
        self.only_label_first_subword = only_label_first_subword
        self.pad_token_label = pad_token_label
        assert self.max_len > 2
        assert self.annotation_type =='bioes'
        self.unk_token = unk_token

        self.label2id_dict, self.id2label = \
                self._generate_labelid_dict()
        self.num_labels = len(self.id2label)
        self.tokens_index = []

    def custom_tokenize(self, text):
        """ The custom tokenize function, when the text is list,
        only lowercase conversion is performed without tokenization.

        Args:
            text (list or str): the input text

        Returns:
            list: token list.
        """
        if isinstance(text, (tuple, list)):
            _tokens = []
            for characters in text:
                if self.do_lower_case and (self.never_split is None or characters not in self.never_split):
                    characters = characters.lower()
                if characters in self.tokenizer.get_vocab():
                    _tokens.append(characters)
                else:
                    _tokens.append(self.unk_token)
            return _tokens
        return [text]

    def _generate_labelid_dict(self):
        """Generate a dictionary that maps input to ID and ID to output."""
        if isinstance(self.label_list, list):
            label_list = self.label_list
        else:
            label_list = json.loads(open(self.label_list, encoding='utf-8').read())['classes'][1:]
        num_classes = len(label_list)
        ignore_id = 4 * num_classes + 1
        id2label_dict = {
            0: 'X',
            ignore_id: 'O'
        }
        for index, category in enumerate(label_list):
            start_label = index + 1
            inside_label = index + 1 + num_classes
            end_label = index + 1 + 2*num_classes
            single_label = index + 1 + 3*num_classes
            #add categories prefixed with B-,I-,E-,S- 
            id2label_dict.update({start_label: 'B-' + category})
            id2label_dict.update({inside_label: 'I-' + category})
            id2label_dict.update({end_label: 'E-' + category})
            id2label_dict.update({single_label: 'S-' + category})
        if self.add_start_end:
            id2label_dict.update({4 * num_classes + 2: '[START]',
            4 * num_classes + 3: '[END]'})
        label2id_dict = {value: key for key, value in id2label_dict.items()}
        return label2id_dict, id2label_dict

    def _labels_convert(self, labels, only_label_first_subword=False):
        """ generate labels for WordPiece tokenizer

        Args:
            labels (list[list or str]): NER labels.
            only_label_first_subword (bool): only works if labels are list[str].
        
        Returns:
             list: the labels for tokenized text.

        Examples:
            [jim,henson,was,a,puppeteer]->[jim,hen,##son,was,a,puppet,##eer] tokens_index=[0,1,1,2,3,4,4]
            [B-name, E-name,  O,  O,   O]->[B-name,I-name,E-name,O, O,  O, O] only_label_first_subword=False
            [B-name, E-name O O O ]     -> [B-name, E-name, IGNORE, O, O, O, IGNORE] only_label_first_subword=True
            [name,0,2]                  -> [name,0,3]
            [name,[0,1]]                -> [name,[0,1,2]]
        """
        tokens_index = self.tokens_index
        if isinstance(labels[0],(tuple,list)):
            #labels type are list[list], which means labels are [[type,[]],] or [[type, start, end]] represation.
            if len(labels[0]) == 3:
                #[type, start, end] represation.
                res = []
                for label in labels:
                    name, start, end = label
                    if end <= start:#invalid entity.
                        continue
                    start_new = tokens_index.index(start)
                    end_new = tokens_index.index(end-1)

                    for i in range(end_new, len(tokens_index)+1):
                        if i == len(tokens_index):
                            key = -1
                        else:
                            key = tokens_index[i]
                        if  key != tokens_index[end_new]:
                            break
                    assert start_new < i, (start_new, end_new, i, tokens_index, start, end)
                    res.append([name, start_new, i])
            else:
                #[type, [index]] represation.
                res = []
                for label in labels:
                    name, range_list = label
                    range_list_new = []
                    for index in range_list:
                        start = tokens_index.index(index)
                        for i in range(start, len(tokens_index)+1):
                            if i == len(tokens_index):
                                key = -1
                            else:
                                key = tokens_index[i]
                            if key != tokens_index[start]:
                                break
                        end = i
                        assert start < end, (start, end, tokens_index)
                        range_list_new += list(range(start,end))
                    res.append([name, range_list_new])
        else:
            #['O','B','I'] represation
            if not only_label_first_subword:
                entities = [[t.to_tuple()[1], t.to_tuple()[2], t.to_tuple()[3]] for t in Tokens(labels, IOBES).entities]
                if entities:
                    entities = self._labels_convert(entities)
                res = ['O']*len(tokens_index)
                for entity in entities:
                    name = entity[0]
                    start = entity[1]
                    end = entity[2]
                    if end - start == 1:
                        res[start] = 'S-'+name
                    else:
                        res[start] = 'B-'+name
                        res[start+1:end-1] = ['I-'+name]*(end-start-2)
                        res[end-1] = 'E-'+name
            else:
                res = ['IGNORE']*len(tokens_index)
                for i, index in enumerate(tokens_index):
                    if i == 0 or index != tokens_index[i-1]:#first subword
                        res[i] = labels[index]
        return res

    def _labels_convert_ori(self, entities, tokens_index, only_label_first_subword=False):
        """ post process, map to word-level lables by tokens_index

        Args:
            entities (list): the entities predicted by the model.
            tokens_index (list): the tokens map index on tokenize process.
            only_label_first_subword (bool): whether only label first subword

        Returns:
            list: the entities corresponding the original text(before tokenize).

        Examples:
            [jim,hen,##son,was,a,puppet,##eer]-> [jim,henson,was,a,puppeteer] tokens_index=[0,1,1,2,3,4,4]
            [name,0,3]                 ->[name,0,2]
            [name,[0,1,2]]             ->[name,[0,1]]
            [B-name, I-name,IGNORE,O,O,O,IGNORE]->[name,[0,1]]
        """
        
        res = []
        if not entities:
            return res
        if entities and isinstance(entities[0], (list, tuple)):
            for entity in entities:
                if len(entity) == 3:
                    name, start, end = entity
                    if end - 1 >= len(tokens_index) or start >= len(tokens_index):
                        continue
                    start_ori = tokens_index[start]
                    end_ori = tokens_index[end-1] + 1
                    res.append([name, start_ori, end_ori])
                else:
                    name, range_list = entity
                    range_list_ori = [tokens_index[index] for index in range_list if index < len(tokens_index)]
                    range_list_ori = list(sorted(list(set(range_list_ori))))#去重
                    res.append([name, range_list_ori])
        else:
            if only_label_first_subword:
                word_level_res = []
                for i, index in enumerate(tokens_index):
                    if i>=len(entities):
                        continue
                    if i == 0 or index != tokens_index[i-1]:#first subword
                        word_level_res.append(entities[i])

                entities = [[t.to_tuple()[1], t.to_tuple()[2], t.to_tuple()[3]] for t in Tokens(word_level_res, IOBES).entities]
                return entities
            else:  
                raise NotImplementedError
        #there may be duplicates after mapping back to the original text.
        final = []
        for entity in res:
            if entity not in final:
                final.append(entity)
        return final

    def convert_text2id(self, results):
        """ Convert token to ids.

        Args:
            results (dict): the input containd tokens list

        Returns:
            dict: corresponding ids
        """
        text = results['tokens']
        cls_token_at_end=self.cls_token_at_end#add [CLS] at end or begin
        cls_token = self.cls_token#cls_token
        cls_token_segment_id = self.cls_token_segment_id#cls token type id
        sep_token = self.sep_token#SEP token
        pad_on_left = self.pad_on_left #pad on left
        pad_token = self.pad_token #pad token
        pad_token_segment_id = self.pad_token_segment_id #pad token type id
        sequence_a_segment_id = self.sequence_a_segment_id
        mask_padding_with_zero = self.mask_padding_with_zero
        tokenizer = self.tokenizer

        tokens = []
        tokens_index = []
        #the indexs of tokens after wordpiece tokenizer
        #eg:[jim,henson,was,a,puppeteer]->[jim,hen,##son,was,a,puppet,##eer] tokens_index=[0,1,1,2,3,4,4]
        for i, word in enumerate(text):
            if self.use_custom_tokenize:
                token = self.custom_tokenize(word)
            else:
                token = tokenizer.tokenize(word)
            tokens.extend(token)
            tokens_index += [i]*len(token)


        self.tokens_index = tokens_index
        special_tokens_count = 2
        if len(tokens) > self.max_len - special_tokens_count:
            print('warning: tokens length %s is longer than max_len, truncate to the maximum length!'%len(tokens))
            tokens = tokens[: (self.max_len - special_tokens_count)]
        #add [SEP]
        tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(tokens)#input mask

        if cls_token_at_end:#add [CLS] at end
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
            input_mask += [1 if mask_padding_with_zero else 0]
        else:#add [CLS] at begin
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
            input_mask = [1 if mask_padding_with_zero else 0] + input_mask

        input_len = len(tokens)
        padding_length = self.max_len - input_len
        if pad_on_left:#pad on left
            tokens = ([pad_token] * padding_length) + tokens
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:#pad on right
            tokens += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

        input_ids = tokenizer.convert_tokens_to_ids(tokens)#convert tokens to ids
        assert len(input_ids) == self.max_len
        assert len(input_mask) == self.max_len
        assert len(segment_ids) == self.max_len
        res = dict(input_ids=input_ids,
                    attention_masks=input_mask,
                    token_type_ids=segment_ids,
                    input_len=input_len,
                    tokens_index=tokens_index)
        return res

    def convert_entity2label(self, labels):
        """ Convert labeled entities to ids.

        Args:
            labels (list): eg:['B-PER', 'I-PER']

        Returns:
            dict: corresponding label ids
        """
        labels = self._labels_convert(labels, self.only_label_first_subword)
        cls_token_at_end=self.cls_token_at_end
        pad_on_left = self.pad_on_left
        pad_token_label = self.pad_token_label
        max_seq_length = self.max_len
        label_map = self.label2id_dict
        label_ids = []
        for x in labels:
            if x in label_map:
                label_ids.append(label_map[x])
            else:
                assert x == 'IGNORE'
                label_ids.append(self.pad_token_label)
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(labels) > max_seq_length - special_tokens_count:
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        label_ids += [label_map['O']]
        if cls_token_at_end:
            label_ids += [label_map['O']]
        else:
            label_ids = [label_map['O']] + label_ids

        # Zero-pad up to the sequence length, padding length is max_length minus label length minus [CLS]和[SEP].
        padding_length = max_seq_length - len(labels) - 2
        if pad_on_left:
            label_ids = ([pad_token_label] * padding_length) + label_ids
        else:
            label_ids += [pad_token_label] * padding_length
        res = dict(labels=label_ids)
        return res

    def convert_pred2entities(self, preds, masks, **kwargs):
        """Gets entities from preds.

        Args:
            preds (list): Sequence of preds.
            masks (tensor): The valid part is 1 and the invalid part is 0.

        Returns:
            list: List of [[[entity_type,
                                entity_start, entity_end]]].
        """
        preds = np.array(preds)[:, 1:].tolist()#remove cls token
        pred_entities = []
        assert isinstance(preds, list)
        for pred in preds:
            tag_list = []
            for tag in pred:
                if not isinstance(tag, str):
                    tag = self.id2label[tag]
                if tag[0] not in ['O', 'B', 'I', 'E', 'S']:
                    tag = 'O'
                tag_list.append(tag)
            if self.only_label_first_subword:
                entities = tag_list
            else:
                entities = [[t.to_tuple()[1], t.to_tuple()[2], t.to_tuple()[3]] for t in Tokens(tag_list, IOBES).entities]
            pred_entities.append(entities)
        tokens_index = [index.cpu().numpy().tolist()[0] for index in kwargs['tokens_index']]
        pred_entities = [self._labels_convert_ori(pred_entity, tokens_index, self.only_label_first_subword) for pred_entity in pred_entities]
        return pred_entities
