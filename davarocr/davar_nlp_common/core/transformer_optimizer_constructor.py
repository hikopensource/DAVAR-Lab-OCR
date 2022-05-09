"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    transformer_optimizer_constructor.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2022-05-06
##################################################################################################
"""
from mmcv.utils import build_from_cfg
from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner import DefaultOptimizerConstructor


@OPTIMIZER_BUILDERS.register_module()
class TransformersOptimizerConstructor(DefaultOptimizerConstructor):
    """Default constructor for transformers optimizers."""

    def __call__(self, model):
        print("****************use transformers constructor*********")
        if hasattr(model, 'module'):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
        no_decay = ["bias", "LayerNorm.weight"]
        bert_parameters = model.encoder.transformers_model.named_parameters()

        decoder_parameters = model.decoder.named_parameters()

        #learning rate use base_lr on bert parameters.
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01, 'lr': self.base_lr},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': self.base_lr}]
        all_encoder_named_paramters = model.encoder.named_parameters()

        #learning rate use 0.001 on non-bert parameters.
        optimizer_grouped_parameters.append({"params": [p for n, p in all_encoder_named_paramters \
            if (not any(nd in n for nd in no_decay)) and 'transformers_model' not in n],
            "weight_decay": 0.01, 'lr': 0.001})
        optimizer_grouped_parameters.append({"params": [p for n, p in all_encoder_named_paramters \
            if any(nd in n for nd in no_decay) and 'transformers_model' not in n],
            "weight_decay": 0.0, 'lr': 0.001})

        optimizer_grouped_parameters.append({"params": [p for n, p in decoder_parameters \
            if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01, 'lr': 0.001})
        optimizer_grouped_parameters.append({"params": [p for n, p in decoder_parameters \
            if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, 'lr': 0.001})
        optimizer_cfg['params'] = optimizer_grouped_parameters

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)


