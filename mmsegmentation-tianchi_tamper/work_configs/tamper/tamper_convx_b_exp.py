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
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=0.6, reduction='none')
            ]
        ),
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
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(type='LovaszLoss', loss_weight=0.6, reduction='none')
            ]
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        ),
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
size = 512
crop_size = 512
ratio = size / crop_size
model['test_cfg'] = dict(mode='slide', stride = (size, size), crop_size = (size, size))
# crop_size = (256, 256)
albu_train_transforms = [
    dict(type='ColorJitter', p=0.5),
    # dict(type='GaussianBlur', p=0.5),
    # dict(type='JpegCompression', p=0.5, quality_lower = 75),
    # dict(type='Affine', rotate=5, shear=5, p=0.5),
    #dict(type='RandomResizedCrop', always_apply = True, height = crop_size, width = crop_size, scale = (0.9, 1.1), ratio = (1.0, 1.0), p=0.5),
    # dict(type='ToGray', p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', cat_max_ratio = 0.75, crop_size = (crop_size, crop_size)),
    dict(type='RandomCopyMove', prob=0.5, mix_prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # dict(type='RandomRotate90', prob=0.5),
    dict(type='Albu', transforms=albu_train_transforms),
    dict(type='Resize', img_scale=(size, size)),
    # dict(type='PhotoMetricDistortion'),
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
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
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
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train2/img',
        ann_dir='train2/msk',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
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
load_from = None # "./work_dirs/tamper/convx_t_8x/epoch_96.pth"
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

nx = 6
total_epochs = int(round(12 * nx))
# optimizer
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg={'decay_rate': 0.99,
                                'decay_type': 'stage_wise',
                                'num_layers': 12})
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(by_epoch=True, interval=total_epochs, save_optimizer=False)
evaluation = dict(by_epoch=True, interval=6, metric=['mIoU', 'mFscore'], pre_eval=True)
fp16 = dict(loss_scale=512.0)

work_dir = f'./work_dirs/tamper/convx_b_exp_{nx}x_lova1_aug1.1_dec1_cpmv'
