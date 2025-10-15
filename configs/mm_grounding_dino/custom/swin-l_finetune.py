_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

load_from = 'weights/grounding_dino_swin-l_pretrain_all-56d69e78.pth'

data_root = 'data/insulator-defect-detection/'
train_ann_file = 'fsod_coco_idx0/train_5shot_seed0.json'
val_ann_file_50 = 'fsod_coco_idx0/val_50.json'
val_ann_file_full = 'fsod_coco_idx0/val.json'
train_dir = 'project/train/img'
val_dir = 'project/val/img'

classes = ('broken', 'insulator', 'pollution-flashover')
num_classes = len(classes)
metainfo = dict(classes=classes, palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)])

num_levels = 5
model = dict(
    use_autocast=True,
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=None),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))),
    bbox_head=dict(num_classes=num_classes))


# --------------------------- dataloader---------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_dir),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file_50,
        data_prefix=dict(img=val_dir)))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file_full,
        data_prefix=dict(img=val_dir)))

val_evaluator = dict(ann_file=data_root + val_ann_file_50)
test_evaluator = dict(ann_file=data_root + val_ann_file_full)

# --------------------------- optimizer ---------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.0005),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))


# learning policy
max_epochs = 300
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=400),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[250],
        gamma=0.1)
]

# hooks
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# custom_hooks = [
#     dict(
#         type='EMAHook',
#         ema_type='ExpMomentumEMA',
#         momentum=0.0002,
#         update_buffers=True,
#         priority=49),
# ]

work_dir = 'work_dirs/swin-l_finetune-bs2-lr2e-5-decay-ema-bert-on-300e (test)'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')