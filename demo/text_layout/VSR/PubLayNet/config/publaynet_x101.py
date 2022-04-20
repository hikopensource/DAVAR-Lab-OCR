# model settings
model = dict(
	type='VSR',
	pretrained=None,
	chargrid=dict(
		type='ChargridEmbedding',
		vocab_size=3500,
		embedding_dim=64,
	),
	sentencegrid=dict(
		type='SentencegridEmbedding',
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

	# visual branch
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
	rpn_head=dict(
		type='RPNHead',
		in_channels=256,
		feat_channels=256,
		anchor_generator=dict(
			type='AnchorGenerator',
			scales=[8],
			ratios=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
			strides=[4, 8, 16, 32, 64]),
		bbox_coder=dict(
			type='DeltaXYWHBBoxCoder',
			target_means=[.0, .0, .0, .0],
			target_stds=[1.0, 1.0, 1.0, 1.0]),
		loss_cls=dict(
			type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
		loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
	roi_head=dict(
		type='CascadeRoIHeadWGCN',
		num_stages=2,
		stage_loss_weights=[1, 0.5],
		bbox_roi_extractor=dict(
			type='SingleRoIExtractor',
			roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
			out_channels=256,
			featmap_strides=[4, 8, 16, 32]),
		bbox_head=[
			dict(
				type='Shared2FCBBoxHeadWGCN',
				in_channels=256,
				fc_out_channels=1024,
				roi_feat_size=7,
				num_classes=6,
				bbox_coder=dict(
					type='DeltaXYWHBBoxCoder',
					target_means=[0., 0., 0., 0.],
					target_stds=[0.1, 0.1, 0.2, 0.2]),
				reg_class_agnostic=True,
				loss_cls=dict(
					type='CrossEntropyLoss',
					use_sigmoid=False,
					loss_weight=1.0),
				loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
				               loss_weight=1.0),
			),
			dict(
				type='Shared2FCBBoxHeadWGCN',
				in_channels=256,
				fc_out_channels=1024,
				roi_feat_size=7,
				num_classes=6,
				bbox_coder=dict(
					type='DeltaXYWHBBoxCoder',
					target_means=[0., 0., 0., 0.],
					target_stds=[0.05, 0.05, 0.1, 0.1]),
				reg_class_agnostic=True,
				loss_cls=dict(
					type='CrossEntropyLoss',
					use_sigmoid=False,
					loss_weight=1.0),
				loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
				               loss_weight=1.0),

				relation_module=dict(
					type='GCNHead',
					in_channels=1024,
					pos_embedding=dict(
						type='PositionEmbedding2D',
						max_position_embeddings=64,
						embedding_dim=1024,
						width_embedding=True,
						height_embedding=True,
					),
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
					with_line_cls=False,
					with_short_cut=True),
			),
		],
		mask_roi_extractor=dict(
			type='SingleRoIExtractor',
			roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
			out_channels=256,
			featmap_strides=[4, 8, 16, 32]),
		mask_head=dict(
			type='FCNMaskHead',
			num_convs=4,
			in_channels=256,
			conv_out_channels=256,
			num_classes=6,
			loss_mask=dict(
				type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg = dict(
        # box level or layout level
        box_level=False,

        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
        ]),
    test_cfg = dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
        )))

# dataset settings
dataset_type = 'PublaynetDataset'
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
		with_cbbox=True,
	    with_cattribute=False,
	    with_ctexts=True,
		with_bbox_2=True,
		with_poly_mask_2=True,
		with_label_2=True,
		# custom_classes=[1, 2, 3],
		# custom_classes_2=[1,2],
		),
	dict(type='DavarResize', img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=32),
	# CharTokenizer for chargrid
	dict(type='CharTokenize', vocab="/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/char_vocab.txt", targets=['gt_ctexts']),
	dict(type='MMLAFormatBundle'),
	dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_texts', 'gt_bboxes_2', 'gt_labels_2', 'gt_masks_2',
	                                 'gt_ctexts', 'gt_cbboxes']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='MMLALoadAnnotations',
	     with_bbox=True,
	     with_text=True,
	     with_label=True,
	     with_care=True,
	     with_poly_mask=False,
	     text_profile=dict(
		     text_max_length=400,
		     sensitive='lower',
		     filtered=False
	     ),
	     with_cbbox=True,
	     with_cattribute=False,
	     with_ctexts=True,
	     with_bbox_2=True,
	     with_poly_mask_2=True,
	     with_label_2=True,
	     # custom_classes=[1, 2, 3],
	     # custom_classes_2=[1,2],
	     ),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(1300, 800),
		flip=False,
		transforms=[
			dict(type='DavarResize', keep_ratio=True),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			# CharTokenizer for chargrid
			dict(type='CharTokenize', vocab="/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/char_vocab.txt",
			     targets=['gt_ctexts']),
			dict(type='MMLAFormatBundle'),
			dict(type='DavarCollect', keys=['img', 'gt_bboxes', 'gt_texts', 'gt_ctexts', 'gt_cbboxes']),
		]
	)
]
data = dict(
	samples_per_gpu=2,
	workers_per_gpu=2,
	train=dict(
		type=dataset_type,
		ann_file=['/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Datalist/datalist_train.json'],
		img_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Images/train/',
		pipeline=train_pipeline,
		ann_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Annos/train/',
		classes=('others', 'text', 'title', 'list', 'table', 'figure')),
	val=dict(
		type=dataset_type,
		ann_file=['/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Datalist/datalist_val.json'],
		img_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Images/dev/',
		pipeline=test_pipeline,
		ann_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Annos/dev/',
		classes=('text', 'title', 'list', 'table', 'figure'),
		coco_ann='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/coco_val.json'),
	test=dict(
		type=dataset_type,
		ann_file=['/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Datalist/datalist_val.json'],
		img_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Images/dev/',
		pipeline=test_pipeline,
		ann_prefix='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/Annos/dev/',
		classes=('text', 'title', 'list', 'table', 'figure'),
		coco_ann='/path/to/Davar-Lab-OCR/demo/text_layout/datalist/PubLayNet/coco_val.json')
	)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 4, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)

checkpoint_config = dict(type='DavarCheckpointHook', interval=1, save_mode='lightweight', metric='bbox_mAP')
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir='/path/to/Davar-Lab-OCR/demo/text_layout/VSR/PubLayNet/log/publaynet_x101/'
load_from = '/path/to/Davar-Lab-OCR/demo/text_layout/VSR/common/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d_with_semantic.pth'
resume_from = None
workflow = [('train', 1)]
evaluation = dict(save_best='bbox_mAP', interval=1, metric='bbox', rule="greater",)
