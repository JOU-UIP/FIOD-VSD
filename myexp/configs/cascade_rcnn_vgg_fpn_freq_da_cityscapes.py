
# 新的配置来自基础的配置以更好地说明需要修改的地方
_base_ = [
    '../../configs/_base_/models/cascade_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/coco_detection.py',
    '../../configs/_base_/schedules/schedule_2x.py',
    '../../configs/_base_/default_runtime.py'
]
# 1. 数据集设定
dataset_type = 'CocoDataset'
classes = ('person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file=(
            '/data/xzr/Datas/cityscapes_series/cityscapes/annotations/train_all.json',
            '/data/xzr/Datas/cityscapes_series/cityscapes_foggy/annotations/train_all_balance.json',
            '/data/xzr/Datas/cityscapes_series/cityscapes_rain/annotations/train_all_balance.json'
                  ),
        img_prefix='/data/xzr/Datas/cityscapes_series/cityscapes_series_Images',
        pipeline=[
            dict(type='LoadImageFromFile',target=2),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1200,600), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','gt_domain_labels','is_source'])
        ]),
    val=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/data/xzr/Datas/cityscapes_series/cityscapes/annotations/val_all.json',
        # ann_file='/data/xzr/Datas/cityscapes_series/cityscapes_foggy/annotations/val_all_balance.json',
        # ann_file='/data/xzr/Datas/cityscapes_series/cityscapes_rain/annotations/val_all_balance.json',
        img_prefix='/data/xzr/Datas/cityscapes_series/cityscapes_series_Images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1200,600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                    ])
        ]),
    test=dict(
        type=dataset_type,
        # 将类别名字添加至 `classes` 字段中
        classes=classes,
        ann_file='/data/xzr/Datas/cityscapes_series/cityscapes/annotations/train_all.json',
        # ann_file='/data/xzr/Datas/cityscapes_series/cityscapes_foggy/annotations/val_all_balance.json',
        # ann_file='/data/xzr/Datas/cityscapes_series/cityscapes_rain/annotations/val_all_balance.json',
        img_prefix='/data/xzr/Datas/cityscapes_series/cityscapes_series_Images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1200,600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1200, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','gt_domain_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Band_stop'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200,600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# 2. 模型设置

model = dict(
    backbone=dict(
        type='Freq_VGG',
        depth=19,
        with_bn=False,
        num_stages=5,
        dilations=(1, 1, 1, 1, 1),
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        bn_eval=False,
        bn_frozen=False,
        ceil_mode=False,
        with_last_pool=False,),
    neck=dict(
        in_channels=[128, 256, 512, 512],),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=8),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=8),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=8),
    ]),
    da_head=dict(
        type='Mult_DAHead_VGG',
        domain_num_classes=3
    )
)
log_config={
    'interval':30,
    'hooks':[
    {
    'type':'TensorboardLoggerHook',
    },
    {
    'type':'TextLoggerHook',
    },
    ]
}
evaluation = dict(interval=1, metric='bbox', save_best='auto')
optimizer = dict(lr=0.01)
checkpoint_config = dict(interval=3)
runner = dict(type='EpochBasedRunner', max_epochs=50)
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0.00001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # step=[16, 19],
    step=[38, 48],
    )
