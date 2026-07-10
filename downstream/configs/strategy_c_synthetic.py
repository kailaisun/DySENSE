"""Strategy C: train on synthetic data only."""
_base_ = ['./strategy_a_real.py']

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='img_dir/train_synthetic',
                         seg_map_path='ann_dir/train_synthetic'),
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='dysense-downstream',
             name='strategy_c_synthetic',
             tags=['strategy_c', 'synthetic_only', 'mask2former_r50', 'multimodal'],
             notes='Strategy C: synthetic training data, real validation.')),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
work_dir = 'output/downstream/strategy_c_synthetic'
