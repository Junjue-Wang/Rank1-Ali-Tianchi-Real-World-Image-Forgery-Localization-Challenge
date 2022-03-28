num_classes = 7

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained="./weights/convnext_large_22k_224.pth",
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[192, 384, 768, 1536], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
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
        in_channels=768,
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
data_root = 'data/binzhou/'
classes = ["x","field", "wood", "grass", "water", "building", "unused"]
palette = [[0,0,0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80]]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], multiply=255., to_rgb=True)
size = 512
# crop_size = (256, 256)
albu_train_transforms = [
    dict(type='ColorJitter', p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged', to_float32=True, max_value_to_uint8=1023),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(size, size)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate90', prob=0.5),
    dict(type='Albu', transforms=albu_train_transforms),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size, size), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged', to_float32=True, max_value_to_uint8=1023),
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
        img_dir='train2/semi_train_GF',
        ann_dir='train2/semi_train_LT',
        img_suffix="_GF.tif",
        seg_map_suffix='_LT.tif',
        classes=classes,
        palette=palette,
        use_mosaic=True,
        mosaic_prob=0.25,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train2/semi_train_GF',
        ann_dir='train2/semi_train_LT',
        img_suffix="_GF.tif",
        seg_map_suffix='_LT.tif',
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
        seg_map_suffix='.tif',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))


# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='CustomizedTextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "./work_dirs/binzhou/convx_l_2x_2lr_mos5v2_aug0_all_t2_512_ft/epoch_24.pth"
# load_from = "./work_dirs/binzhou/convx_l_5x_2lr_mos5v2_aug0_all_t2_384/epoch_60.pth"
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

nx = 10
total_epochs = int(round(12 * nx))
# optimizer
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=0.0001 / 4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.9,
                                'decay_type': 'stage_wise',
                                'num_layers': 12,
                                # 'custom_keys': {
                                #     'head': dict(lr_mult=10.)
                                # }
                            })
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(by_epoch=True, interval=12)
evaluation = dict(by_epoch=True, interval=12, metric='mIoU', pre_eval=True)
fp16 = dict(loss_scale=512.0)

work_dir = f'./work_dirs/binzhou/convx_l_{nx}x_2lr_mos25v2_aug0_all_t2_512_ft2'
