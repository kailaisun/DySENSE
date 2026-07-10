"""Strategy B: train on real + synthetic (mix)."""
_base_ = ['./strategy_a_real.py']

data_root = 'data/muse_mmseg_class10'
TRAIN_JSONL = 'data/MUSE2/mixed_json_class10_july/train-4cities-class10-july.jsonl'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs',
         meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'reduce_zero_label', 'climate_text')),
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='MUSEEnergyPromptDataset10',
                prompt_jsonl=TRAIN_JSONL,
                prompt_mode='correct',
                data_root=data_root,
                data_prefix=dict(img_path='img_dir/train_real',
                                 seg_map_path='ann_dir/train_real'),
                pipeline=train_pipeline,
            ),
            dict(
                type='MUSEEnergyPromptDataset10',
                prompt_jsonl=TRAIN_JSONL,
                prompt_mode='correct',
                data_root=data_root,
                data_prefix=dict(img_path='img_dir/train_synthetic',
                                 seg_map_path='ann_dir/train_synthetic'),
                pipeline=train_pipeline,
            ),
        ],
    ),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=5000)
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=0, end=60000, by_epoch=False)
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000,
                    max_keep_ckpts=3, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='dysense-downstream',
             name='strategy_b_mix',
             tags=['strategy_b', 'real_plus_synthetic', 'mask2former_r50', 'multimodal'],
             notes='Strategy B: real + synthetic training data, real validation.')),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
work_dir = 'output/downstream/strategy_b_mix'
