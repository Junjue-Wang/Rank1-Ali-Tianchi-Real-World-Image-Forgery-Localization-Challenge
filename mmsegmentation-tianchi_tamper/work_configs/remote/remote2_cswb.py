num_classes = 47

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained="./weights/cswin_base_384.pth",
    backbone=dict(
        type='CSWin',
        in_chans=4,
        embed_dim=96,
        patch_size=4,
        depth=[2,4,32,2],
        num_heads=[4,8,16,32],
        split_size=[1,2,7,7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.6,
        use_chk=False),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96,192,384,768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/remote2/'
classes = ["background", "dry", "fruit", "tea", "mulberry", "latex", "nursery", "flower", "other", "trees", "woods", "tree_wood", "banboo", "forest", "greening", "artificial_trees", "sparse_woods", "natural_grass", "artificial_grass", "multi_building", "low_building", "abandoned_building", "multi_independent", "low_independent", "road", "railway", "hard_ground", "water_structure", "wall", "greenhouse", "curing_pool", "industry", "sand_obstacle", "other_building", "outdoor_mining", "piles", "construction", "other_artificial", "salt_land", "mud_land", "sand_land", "dirt_land", "stone_land", "cannels", "water", "ice_snow", "water_field"]
palette = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 114.50], std=[58.395, 57.12, 57.375, 57.63], to_rgb=True)
size = 512
crop_size = (size, size)
albu_train_transforms = [
    dict(type='RandomRotate90', p=0.5),
    dict(type='GridDistortion', p=0.5),
]
train_pipeline = [
    dict(type='LoadImageAlphaFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(512, 512), ratio_range=(1, 1)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # dict(type='RandomRotate90', prob=0.5),
    # dict(type='Albu', transforms=albu_train_transforms),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageAlphaFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(size, size),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/labels',
        img_suffix=".tif",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/labels',
        img_suffix=".tif",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        img_dir='test/images',
        ann_dir='test/labels',
        img_suffix=".tif",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))


# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = "./weights/segformer_mit-b4_512x512_160k_ade20k_20210728_183055-7f509d7d.pth"
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

total_epochs = 12
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(by_epoch=True, interval=12)
evaluation = dict(by_epoch=True, interval=12, metric='mIoU', pre_eval=True)
fp16 = dict(loss_scale=512.0)

work_dir = './work_dirs/remote2/cswb384_1x_16bs_all'
