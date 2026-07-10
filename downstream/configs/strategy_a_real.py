"""Strategy A: train on real data only."""

custom_imports = dict(
    imports=['mmseg_custom.datasets', 'mmseg_custom.models'],
    allow_failed_imports=False)

REMOTECLIP_CKPT = 'checkpoints/RemoteCLIP-ViT-L-14.pt'
TRAIN_JSONL = 'data/MUSE2/mixed_json_class10_july/train-4cities-class10-july.jsonl'
TEST_JSONL = 'data/MUSE2/mixed_json_class10_july/test-4cities-class10-july.jsonl'

data_root = 'data/muse_mmseg_class10'

num_classes = 9

model = dict(
    decode_head=dict(
        type='Mask2FormerMultimodalHead',
        remoteclip_ckpt=REMOTECLIP_CKPT,
        num_classes=num_classes,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1],
        ),
    ),
)

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

META_KEYS = ('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape',
             'scale_factor', 'flip', 'flip_direction', 'reduce_zero_label',
             'climate_text')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=META_KEYS),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs', meta_keys=META_KEYS),
]

train_dataloader = dict(
    batch_size=30,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MUSEEnergyPromptDataset10',
        prompt_jsonl=TRAIN_JSONL,
        prompt_mode='correct',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train_real',
                         seg_map_path='ann_dir/train_real'),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=30,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MUSEEnergyPromptDataset10',
        prompt_jsonl=TEST_JSONL,
        prompt_mode='correct',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'remoteclip': dict(lr_mult=0.1),
            'norm': dict(decay_mult=0.0),
            'query_embed': dict(decay_mult=0.0),
            'query_feat': dict(decay_mult=0.0),
        }
    ),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=0, end=40000, by_epoch=False)
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000,
                    max_keep_ckpts=3, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='dysense-downstream',
             name='strategy_a_real',
             tags=['strategy_a', 'real_only', 'mask2former_r50', 'multimodal'],
             notes='Strategy A: real training data, real validation.')),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
work_dir = 'output/downstream/strategy_a_real'
