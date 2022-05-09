"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    base_ner.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
import numpy as np
from davarocr.davar_nlp_common.models import EncoderDecoder
from ..builder import NERS


@NERS.register_module()
class BaseNER(EncoderDecoder):
    """Base class for NER classifier."""
    def get_encoder_out(self, input_ids, attention_masks=None, token_type_ids=None, **kwargs):
        """ Get encoder's output, note the text range's intersection must be empty.
        """
        assert "range" in kwargs
        assert len(input_ids) == len(attention_masks)
        assert len(input_ids) == len(token_type_ids)
        assert len(input_ids) == len(kwargs['range'])
        max_text_length = kwargs['range'][-1].cpu().numpy().ravel()[-1]

        encoder_output = []
        all_token_index = []
        for i, (input_id, attention_mask, token_type_id) in \
        enumerate(zip(input_ids, attention_masks, token_type_ids)):#sliding window test's input
            other_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, list) and len(value) == len(input_ids):
                    other_kwargs[key] = value[i]
            token_range = list(kwargs['range'][i].cpu().numpy().ravel())
            start, end = token_range
            tokens_index = [index.cpu().numpy().tolist()[0] for index in other_kwargs['tokens_index']]
            tokens_index = [start + index for index in tokens_index]
            encode_out = self.encoder(input_ids=input_id,
                                    attention_masks=attention_mask,
                                    token_type_ids=token_type_id,
                                    **other_kwargs)
            all_token_index += tokens_index
            encoder_output.append(encode_out[0][0].cpu().numpy()[1:1+len(tokens_index)])
        encoder_output = np.vstack(encoder_output)
        return encoder_output, all_token_index

    def aug_test(self, input_ids, attention_masks=None, token_type_ids=None, **kwargs):
        """Aug test. currently only support batch is 1 and the text range's intersection is empty.
        used for sliding window test.
        """
        assert "range" in kwargs
        assert len(input_ids) == len(attention_masks)
        assert len(input_ids) == len(token_type_ids)
        assert len(input_ids) == len(kwargs['range'])
        range_list = kwargs['range']
        res = []
        for i, (input_id, attention_mask, token_type_id) in \
        enumerate(zip(input_ids, attention_masks, token_type_ids)):#sliding window test's input
            other_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, list) and len(value) == len(input_ids):
                    other_kwargs[key] = value[i]
            pred_entities = self.simple_test(input_id, attention_mask, token_type_id,**other_kwargs)
            for batch_index, batch_entity in enumerate(pred_entities):#batch must be one
                temp_entity = []
                for _entity in batch_entity:
                    if len(_entity) == 3:
                        entity = [_entity[0],
                                int(_entity[1] + range_list[i].cpu().numpy()[batch_index][0]),\
                                int(_entity[2] + range_list[i].cpu().numpy()[batch_index][0])]#add offset
                    else:
                        entity = [_entity[0],
                                list(map(lambda x: x + range_list[i].cpu().numpy()[batch_index][0], _entity[1]))]
                    temp_entity.append(entity)
                res += temp_entity
        return res
