# model settings
model = dict(
    type='ChargridNetLayout',
    use_chargrid=True,  # for testing raw image input, set to False
    is_cat=True,    # for testing raw image input, set to False
    pretrained=None,
    embedding=dict(
        type='Embedding',
        vocab_size=96,
        embedding_dim=64,
    ),
    backbone=dict(
        type='ChargridEncoder',
        input_channels=99,  # for testing raw image input, set to 3
        out_indices=(0, 1, 2, 4),
        base_channels=64),
    neck=dict(
        type='ChargridDecoder',
        base_channels=64),
    rpn_head=dict(
        type='RPNHead',
        in_channels=64,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='ChargridRegHead',
            in_channels=64,
            conv_out_channels=64,
            roi_feat_size=7,
            num_classes=6,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=3,
            in_channels=64,
            conv_out_channels=64,
            num_classes=6,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),

    # model training and testing settings
    train_cfg=dict(
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
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=100,
            mask_thr_binary=0.5)
    )
)

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
		with_label=False,
		with_poly_mask=False,
		text_profile=dict(
			text_max_length=400,
			sensitive='lower',
			filtered=False
		),
		with_cbbox=False,
	    with_cattribute=False,
	    with_ctexts=False,
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
	dict(type='ChargridDataGeneration', with_label=False, poly_shape=False),
	dict(type='ChargridFormatBundle'),
	dict(type='DavarCollect', keys=['img', 'chargrid_map', 'gt_bboxes_2', 'gt_labels_2', 'gt_masks_2']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='MMLALoadAnnotations',
	     with_bbox=True,
	     with_text=True,
	     with_label=False,
	     with_care=True,
	     with_poly_mask=False,
	     text_profile=dict(
		     text_max_length=400,
		     sensitive='lower',
		     filtered=False
	     ),
	     with_cbbox=False,
	     with_cattribute=False,
	     with_ctexts=False,
	     with_bbox_2=False,
	     with_poly_mask_2=False,
	     with_label_2=False,
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
            dict(
                type='ChargridDataGeneration',
                # visualize=True,
                # vis_save_dir='/path/to/demo/text_ie/chargrid/vis/publaynet/chargrid',
                with_label=False,
                poly_shape=False),
            dict(type='ChargridFormatBundle'),
			dict(type='DavarCollect', keys=['img', 'chargrid_map']),
		]
	)
]
data = dict(
	samples_per_gpu=4,
	workers_per_gpu=2,
	train=dict(
		type=dataset_type,
		ann_file='/path/to/demo/text_ie/datalist/PubLayNet/sampled_datalist_train_new.json',
		img_prefix='/path/to/PubLayNet/',
        ann_prefix='/path/to/PubLayNet/Annos/train/',
		pipeline=train_pipeline,
		classes=('others', 'text', 'title', 'list', 'table', 'figure')),
	val=dict(
		type=dataset_type,
		ann_file='/path/to/demo/text_ie/datalist/PubLayNet/sampled_datalist_val_new.json',
		img_prefix='/path/to/PubLayNet/',
        ann_prefix='/path/to/PubLayNet/Annos/dev/',
		pipeline=test_pipeline,
		classes=('text', 'title', 'list', 'table', 'figure'),
		coco_ann='/path/to/PubLayNet/Datalist/coco_val.json'),
	test=dict(
		type=dataset_type,
		ann_file='/data1/repo/demo/text_ie/datalist/PubLayNet/sampled_datalist_val_new.json',
		img_prefix='/path/to/PubLayNet/',
        ann_prefix='/path/to/PubLayNet/Annos/dev/',
        pipeline=test_pipeline,
		classes=('text', 'title', 'list', 'table', 'figure'),
		coco_ann='/path/to/PubLayNet/Datalist/coco_val.json')
	)
# optimizer
find_unused_parameters = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[15, 30, 40])
runner = dict(type='EpochBasedRunner', max_epochs=50)

checkpoint_config = dict(type='DavarCheckpointHook', interval=1, save_mode='lightweight', metric='bbox_mAP')
# yapf:disable
log_config = dict(
    interval=40,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir='/path/to/demo/text_ie/chargrid/log/publaynet_chargrid/'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(save_best='bbox_mAP', interval=1, metric='bbox', rule="greater",)