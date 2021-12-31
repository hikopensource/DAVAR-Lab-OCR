model = dict(
    type='VSR',
    pretrained=None,
	bertgrid=dict(
		type='BERTgridEmbedding',
		auto_model_path="/path/to/Davar-Lab-OCR/demo/text_layout/VSR/common/bert-base-uncased/",
		embedding_dim=64,
	),

	# multimodal merge
	multimodal_merge=dict(
		with_img_feat=False,

		multimodal_feat_merge=dict(
			type='VSRFeatureMerge',
			merge_type='Weighted',
			visual_dim=[256, 512, 1024, 2048],
			semantic_dim=[256, 512, 1024, 2048],
			with_extra_fc=False,
		),
	),

	# semantic branch
	backbone_semantic=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),

    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    # line GCN
	line_roi_extractor=dict(
		type='SingleRoIExtractor',
		roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
		out_channels=256,
		featmap_strides=[4, 8, 16, 32]),
	line_gcn_head=dict(
		type='GCNHead',
		in_channels=256,
		num_fcs=2,
		fc_out_channels=1024,
		roi_feat_size=7,
		with_line_cls=True,
		num_classes=13,
		loss_line_cls=dict(
			type='StandardCrossEntropyLoss',
			ignore_index=255),

		gcn_module=dict(
			type='BertEncoder',
			config=dict(
				hidden_size=1024,
				num_hidden_layers=2,
				num_attention_heads=16,
				intermediate_size=1024,  # 4 x hidden_size
				hidden_act="gelu",
				hidden_dropout_prob=0.1,
				attention_probs_dropout_prob=0.1,
				layer_norm_eps=1e-12,
				output_attentions=False,
				output_hidden_states=False,
				is_decoder=False, )),
	),
	# model training and testing settings
	train_cfg = dict(),
	test_cfg = dict()
)

# dataset settings
dataset_type = 'DocBankDataset'
data_root = ''
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='MMLALoadAnnotations',
		with_care=True,
		with_bbox=True,
		with_text=True,
		with_label=True,
		with_poly_mask=False,
		text_profile=dict(
			text_max_length=400,
			sensitive='lower',
			filtered=False
		),
		 with_cbbox=False,
		 with_cattribute=False,
		 with_ctexts=False,
        label_start_index=0
		),
	dict(type='DavarResize', img_scale=(600, 800)),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	dict(type='MMLAFormatBundle'),
	dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_texts']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='MMLALoadAnnotations',
	     with_care=True,
	     with_bbox=True,
	     with_text=True,
	     with_label=True,
	     with_poly_mask=False,
	     text_profile=dict(
		     text_max_length=400,
		     sensitive='lower',
		     filtered=False
	     ),
	     with_cbbox=False,
	     with_cattribute=False,
	     with_ctexts=False,
	     label_start_index=0
	     ),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(600, 800),
		flip=False,
		transforms=[
			dict(type='DavarResize', keep_ratio=True),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			dict(type='MMLAFormatBundle'),
			dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_texts']),
		]
	)
]
data = dict(
	samples_per_gpu=2,
	workers_per_gpu=0,
	train=dict(
		type=dataset_type,
		ann_file='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Datalist/500K_train_datalist.json',
		img_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Images/',
		pipeline=train_pipeline,
		ann_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Annos/',
        classes_config='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Datalist/classes_config.json',
		classes=('abstract', 'author', 'caption', 'date', 'equation', 'figure', 'footer', 'list', 'paragraph',
				 'reference', 'section', 'table', 'title'),
	    max_num=2048),
	val=dict(
		type=dataset_type,
        ann_file='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Datalist/500K_dev_datalist.json',
        img_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Images/',
		pipeline=test_pipeline,
        ann_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Annos/',
        classes_config='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Datalist/classes_config.json',
        classes=(
        'abstract', 'author', 'caption', 'date', 'equation', 'figure', 'footer', 'list', 'paragraph', 'reference',
        'section', 'table', 'title'),
		max_num=2048),
	test=dict(
		type=dataset_type,
        ann_file='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Datalist/500K_test_datalist.json',
        img_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Images/',
		pipeline=test_pipeline,
        ann_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Annos/',
        classes_config='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/DocBank/Datalist/classes_config.json',
        classes=(
        'abstract', 'author', 'caption', 'date', 'equation', 'figure', 'footer', 'list', 'paragraph', 'reference',
        'section', 'table', 'title'),
		max_num=204800))
# optimizer
# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam',  amsgrad=False, betas=(0.9, 0.999), eps=1e-8, lr=0.00001, weight_decay=0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.001,
    step=[2, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)

checkpoint_config = dict(type='DavarCheckpointHook', interval=1, save_mode='lightweight', metric='avg_f1')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir='/path/to/Davar-Lab-OCR/demo/text_layout/VSR/DocBank/log/docbank_x101/'
load_from = '/path/to/Davar-Lab-OCR/demo/text_layout/VSR/common/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d_with_semantic.pth'
resume_from = None
workflow = [('train', 1)]
evaluation = dict(save_best='avg_f1', interval=1, metric='F1-score', rule="greater",)
