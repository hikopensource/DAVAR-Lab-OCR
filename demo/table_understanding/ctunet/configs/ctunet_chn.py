# model settings
batch_max_length = 60
type = "SPOTTER"
model = dict(
    type='CTUNet',
    pretrained='path/to/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
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
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=5,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    infor_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=(1, 1), sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    raw_text_embedding=dict(
        type='Embedding',
        vocab_size=8500,
        embedding_dim=256,
    ),
    infor_context_module=dict(
        type='MultiModalContextModulePlusPlus',
        mode=1,
        with_visual=True,
        textual_embedding=dict(
            type='TextualEmbedding',
            dropout_ratio=0.1,
            merge_type='SumNew',
            sentence_embedding=dict(
                type='SentenceEmbeddingCNN',
                embedding_dim=256,
                kernel_sizes=[3, 5, 7, 9]
            ),
            sentence_embedding_bert=dict(
                type='SentenceEmbeddingBertNew',
                auto_model_path='/path/to/bert-base-chinese/',
                embedding_dim=768,
                freeze_params=True,
                character_wise=True,
                use_cls=True,
                batch_max_length=batch_max_length
            ),
        ),
        pos_embedding=dict(
            type='PositionEmbedding2D',
            max_position_embeddings=64,
            embedding_dim=256,
            width_embedding=True,
            height_embedding=True,
        ),
        relative_pos_embedding=dict(
            type='RelativePositionEmbedding2D',
            max_position_embeddings=100,
            embedding_dim=8,
        ),
        textual_relation_module=dict(
            type='BertEncoder',
            config=dict(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=32,
                intermediate_size=512,  # 4 x hidden_size
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                layer_norm_eps=1e-12,
                output_attentions=False,
                output_hidden_states=False,
                is_decoder=False, )
        )),
    infor_node_cls_head=dict(
        type='TableClsHead',
        input_size=256,
        num_classes=5,  #
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False)
    ),
    infor_row_relation_head=dict(
        type='TableClsHead',
        input_size=512,
        fc_out_channels=1024,
        num_fcs=3,
        num_classes=2,  #
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False)
    ),
    infor_col_relation_head=dict(
        type='TableClsHead',
        input_size=512,
        fc_out_channels=1024,
        num_fcs=3,
        num_classes=2,  #
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False)
    ),

    # model training and testing settings
    train_cfg=dict(
        # rcg
        keep_dim=False,
        sequence=(),

        # infor
        down_sample_first=False,

        reproduce_labels=False,
        batch_max_length=batch_max_length,

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
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        # rcg
        keep_dim=False,
        sequence=dict(),
        batch_max_length=batch_max_length,

        # info
        down_sample_first=False,

        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

# dataset settings
dataset_type = 'CTUNetDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='CTUNetLoadAnnotations',
         with_bbox=True,  # Bounding Rect
         with_poly_mask=False,  # Mask
         with_poly_bbox=False,  # bouding poly
         with_label=True,  # Bboxes' labels
         with_care=True,  # Ignore or not
         with_text=True,  # Transcription
         with_cbbox=False,  # Character bounding
         text_profile=dict(text_max_length=batch_max_length, sensitive='same', filtered=False),
         with_relations=True,
         with_rowcols=True
         ),
    # dict(type='DavarResize', img_scale=[(512, 512)], keep_ratio=True, multiscale_mode='range'),
    dict(type='DavarResize', img_scale=[(320, 420), (960, 1440)], keep_ratio=True, multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='CharPadTokenize', vocab=None, targets=['gt_texts'], max_length=batch_max_length,
         map_target_prefix='array_'),
    dict(type='CTUNetFormatBundle'),
    dict(type='DavarCollect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_texts', 'array_gt_texts', 'relations', 'gt_rowcols']),
]
val_pipeline = [
    dict(type='DavarLoadImageFromFile'),
    dict(type='CTUNetLoadAnnotations',
         with_bbox=True,  # Bounding Rect
         with_poly_mask=False,  # Mask
         with_poly_bbox=False,  # bouding poly
         with_label=True,  # Bboxes' labels
         with_care=True,  # Ignore or not
         with_text=True,  # Transcription
         with_cbbox=False,  # Character bounding
         text_profile=dict(text_max_length=batch_max_length, sensitive='same', filtered=False),
         with_relations=True,
         with_rowcols=True
         ),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(512, 512),
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='CharPadTokenize', vocab=None, targets=['gt_texts'], max_length=batch_max_length,
                 map_target_prefix='array_'),
            dict(type='CTUNetFormatBundle'),
            dict(type='DavarCollect',
                 keys=['img', 'gt_bboxes', 'gt_labels', 'gt_texts', 'array_gt_texts', 'relations', 'gt_rowcols']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file="path/to/ComFinTab_chn_train.json",
        img_prefix="path/to/ComFinTab/chn/Images/train_img",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file="path/to/ComFinTab_chn_test.json",
        img_prefix="path/to/ComFinTab/chn/Images/test_img",
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file="path/to/ComFinTab_chn_test.json",
        img_prefix="path/to/ComFinTab/chn/Images/test_img",
        pipeline=val_pipeline
    ))
# optimizer
find_unused_parameters = True
optimizer = dict(type='AdamW', lr=1e-3)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='inv', warmup='linear', warmup_iters=10,
                 warmup_ratio=0.00001, warmup_by_epoch=True, gamma=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=50)

checkpoint_config = dict(type='DavarCheckpointHook', interval=2, save_mode='lightweight', metric='tree_f1'
                         , filename_tmpl='checkpoint/cell_cls_edge{}.pth', save_last=True)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'path/to/work_dir'
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(
    save_best='tree_f1',
    model_type='SPOTTER',
    interval=2,
    metric='tree_f1',
    rule="greater",
    metric_options=dict(
        macro_f1=dict(),
        save_result=False,
        save_prefix=work_dir,  # where to save the inference results
        save_name='ComFinTab_chn_test_pred.json'),  # where to save the inference results
    priority='HIGH')
