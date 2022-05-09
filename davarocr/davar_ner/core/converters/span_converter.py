"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    span_converter.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
from seqeval.scheme import Tokens, IOBES
from davarocr.davar_common.core import CONVERTERS
from .transformers_converter import TransformersConverter


@CONVERTERS.register_module()
class SpanConverter(TransformersConverter):
    """Span converter, converter for span model.
    """
    def _generate_labelid_dict(self):
        label2id_dict = {label: i for i, label in enumerate(['O'] + self.label_list)}
        id2label_dict = {value: key for key, value in label2id_dict.items()}
        return label2id_dict, id2label_dict


    def _extract_subjects(self, seq):
        """Get entities from label sequence
        """
        entities = [(t.to_tuple()[1], t.to_tuple()[2], t.to_tuple()[3]) for t in Tokens(seq, IOBES).entities]
        return entities


    def convert_entity2label(self, labels):
        """Convert labeled entities to ids.

        Args:
            labels (list): eg:['B-PER', 'I-PER']

        Returns:
            dict: corresponding ids
        """
        labels = self._labels_convert(labels, self.only_label_first_subword)
        cls_token_at_end=self.cls_token_at_end
        pad_on_left = self.pad_on_left
        label2id = self.label2id_dict
        subjects = self._extract_subjects(labels)#get entities
        start_ids = [0] * len(labels)
        end_ids = [0] * len(labels)
        subjects_id = []
        for subject in subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]

            #set label for span
            start_ids[start] = label2id[label]
            end_ids[end-1] = label2id[label]#the true position is end-1
            subjects_id.append((label2id[label], start, end))

        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(labels) > self.max_len - special_tokens_count:
            start_ids = start_ids[: (self.max_len - special_tokens_count)]
            end_ids = end_ids[: (self.max_len - special_tokens_count)]

        #add sep
        start_ids += [0]
        end_ids += [0]
        if cls_token_at_end:
            #add [CLS] at end
            start_ids += [0]
            end_ids += [0]
        else:
            #add [CLS] at begin
            start_ids = [0]+ start_ids
            end_ids = [0]+ end_ids
        padding_length = self.max_len - len(labels) - 2
        if pad_on_left:
            #pad on left
            start_ids = ([0] * padding_length) + start_ids
            end_ids = ([0] * padding_length) + end_ids
        else:
            #pad on right
            start_ids += ([0] * padding_length)
            end_ids += ([0] * padding_length)
        res = dict(start_positions=start_ids, end_positions=end_ids)
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
        id2label = self.id2label
        pred_entities = []
        for pred in preds:
            entities = []
            entity = [0, 0, 0]
            for tag in pred:
                entity[0] = id2label[tag[0]]
                entity[1] = tag[1] - 1
                entity[2] = tag[2] - 1
                entities.append(entity.copy())
            pred_entities.append(entities.copy())
        tokens_index = [index.cpu().numpy().tolist()[0] for index in kwargs['tokens_index']]
        pred_entities = [self._labels_convert_ori(pred_entity, tokens_index) for pred_entity in pred_entities]
        return pred_entities
