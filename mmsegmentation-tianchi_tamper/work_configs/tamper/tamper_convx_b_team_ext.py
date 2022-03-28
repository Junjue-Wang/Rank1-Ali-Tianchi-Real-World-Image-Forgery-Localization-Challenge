num_classes = 2

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained="./weights/convnext_base_22k_224.pth",
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[128, 256, 512, 1024], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.75),
        #     dict(type='DiceLoss', loss_weight=0.25)
        #     ]
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=True)
            ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=1.0, per_image=True)
            ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())

# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/tamper/'
classes = ["0", "1"]
palette = [[0,0,0], [255,255,255]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
size = 1024
crop_size = 1024
ratio = 2.0
model['test_cfg'] = dict(mode='slide', stride = (size, size), crop_size = (size, size))
albu_train_transforms = [
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', ratio_range=(1.0, 3.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(crop_size, crop_size), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[ratio],
        flip=False,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/img',
            ann_dir='train/msk',
            img_suffix=".jpg",
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            use_mosaic=False,
            mosaic_prob=0.5,
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train2/img',
            ann_dir='train2/msk',
            img_suffix=".jpg",
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            use_mosaic=False,
            mosaic_prob=0.5,
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='ext1/img',
            ann_dir='ext1/msk',
            img_suffix=".jpg",
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            use_mosaic=False,
            mosaic_prob=0.5,
            pipeline=train_pipeline),
        ],
    val=[
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/img',
            ann_dir='train/msk',
            test_mode=True,
            img_suffix=".jpg",
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            use_mosaic=False,
            mosaic_prob=0.5,
            pipeline=test_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train2/img',
            ann_dir='train2/msk',
            test_mode=True,
            img_suffix=".jpg",
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            use_mosaic=False,
            mosaic_prob=0.5,
            pipeline=test_pipeline)
        ],
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        img_dir='test/img',
        ann_dir='test/msk',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "./work_dirs/tamper/convx_b_team_ft2_30k/latest.pth"
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

total_iters = 20
# optimizer
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings

runner = dict(type='IterBasedRunner', max_iters=total_iters * 1000)
checkpoint_config = dict(by_epoch=False, interval=5 * 1000, save_optimizer=False)
evaluation = dict(by_epoch=False, interval=10 * 1000, metric=['mIoU', 'mFscore'], pre_eval=True)
fp16 = dict()

work_dir = f'./work_dirs/tamper/convx_b_team_ext_{total_iters}k'
