"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    bert_crf.py
# Abstract       :    BERT+CRF for NER task.

# Current Version:    1.0.0
# Date           :    2022-02-23
##################################################################################################
"""
_base_ = [
    '../base/default_runtime.py'
]
"""
1. Data Setting
description:
    Pipeline and training dataset settings

Add keywords:
    None
"""
dataset_name = 'conll2003'
if dataset_name == 'conll2003':
    test_ann_file = '/data1/data/English/Flat/CoNLL2003/Datalist/test.json'
    train_ann_file = ['/data1/data/English/Flat/CoNLL2003/Datalist/train.json',
                      '/data1/data/English/Flat/CoNLL2003/Datalist/dev.json']
    val_ann_file = '/data1/data/English/Flat/CoNLL2003/Datalist/test.json'
    label_list = ['ORG', 'MISC', 'PER', 'LOC']
    model_name_or_path = '/data1/resume/models/huggingface/bert-base-cased'
    do_lower_case = False
    #dataset label list
if dataset_name == 'RESUME':
    train_ann_file = '/data1/data/Chinese/Flat/RESUME/Datalist/train.json'
    val_ann_file = '/data1/data/Chinese/Flat/RESUME/Datalist/test.json'
    test_ann_file = '/data1/data/Chinese/Flat/RESUME/Datalist/test.json'
    label_list = ['NAME', 'CONT', 'RACE', 'TITLE', 'EDU', 'ORG', 'PRO', 'LOC']
    model_name_or_path = '/data1/resume/models/huggingface/bert-base-chinese'
    do_lower_case = True
work_dir = './work_dirs/bert_crf_%s'%dataset_name

max_len=256

loader = dict(
    type='NERLoader',
	truncation=False,
	stride=max_len-2,
	max_len=max_len-2)

ner_converter = dict(
    type='TransformersConverter',
    model_name_or_path=model_name_or_path,
	label_list = label_list,
	max_len = 512,
    do_lower_case=do_lower_case,
    )

test_pipeline = [
    dict(type='NERTransform', label_converter=ner_converter),
    dict(type='ToTensor',keys=['input_ids', 'attention_masks', "token_type_ids", "labels", "input_len"])
]

train_pipeline = [
    dict(type='NERTransform', label_converter=ner_converter),
    dict(type='ToTensor',keys=['input_ids', 'attention_masks', "token_type_ids", "labels", "input_len"])
]
dataset_type = 'NERDataset'

train = dict(
    type=dataset_type,
    ann_file=train_ann_file,
    loader=loader,
    pipeline=train_pipeline,
    test_mode=False)

val = dict(
    type=dataset_type,
    ann_file=val_ann_file,
    loader=loader,
    pipeline=test_pipeline,
    test_mode=True)
test = dict(
    type=dataset_type,
    ann_file=test_ann_file,
    loader=loader,
    pipeline=test_pipeline,
    test_mode=True)
data = dict(
    samples_per_gpu=4, workers_per_gpu=2, train=train, val=val, test=test)

"""
2. model setting
description:
    NER model configuration information

Add keywords:
    None
"""
type = 'NER'
model = dict(
    type='BaseNER',
	encoder=dict(type='TransformersEncoder',model_name_or_path=model_name_or_path),
	decoder=dict(type='CRFDecoder',label_converter=ner_converter)
	)

test_cfg = None
"""
3. Training parameter settings
description:
    Configure the corresponding learning rate and related strategy according to the dataset or model structure

Add keywords:
    None

"""
# optimizer
#optimizer = dict(type='SGD', lr=6e-4, momentum=0.9)
optimizer = dict(type='AdamW', lr=5e-5, constructor='TransformersOptimizerConstructor')
optimizer_config = dict(grad_clip=dict(max_norm=5))
# learning policy
lr_config = dict(policy='inv', warmup='linear', warmup_iters=10,
    warmup_ratio=0.00001,warmup_by_epoch=True, gamma=0.05)
total_epochs = 50
find_unused_parameters=True
"""
4. Evaluation and checkpoint settings
"""
evaluation = dict(interval=1, metric='f1-score',save_best='hmean')
checkpoint_config = dict(type='DavarCheckpointHook', interval=1, save_mode='lightweight', metric='hmean',
                         filename_tmpl='bert_e{}.pth', save_last=False)
