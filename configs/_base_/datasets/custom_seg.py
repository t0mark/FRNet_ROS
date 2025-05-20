# For SemanticKitti we usually do 19-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'MLDAS'
data_root = 'data/MLDAS/'
class_names = [
    'unlabel', 'road', 'sidewalk', 'car', 'other-vehicle',
]
labels_map = {
    0: 0,   # unlabeled                   -> 0: unlabeled
    1: 0,   # outlier                     -> 0: unlabeled
    10: 3,  # car                         -> 3: car
    11: 4,  # bicycle                     -> 4: other-vehicle
    13: 3,  # bus                         -> 3: car
    15: 4,  # motorcycle                  -> 4: other-vehicle
    16: 4,  # on-rails                    -> 4: other-vehicle
    18: 3,  # truck                       -> 3: car
    20: 4,  # other-vehicle               -> 4: other-vehicle
    30: 4,  # person                      -> 4: other-vehicle
    31: 4,  # bicyclist                   -> 4: other-vehicle
    32: 4,  # motorcyclist                -> 4: other-vehicle
    40: 1,  # road                        -> 1: road
    44: 1,  # parking                     -> 1: road
    48: 2,  # sidewalk                    -> 2: sidewalk
    49: 0,  # other-ground                -> 0: unlabeled
    50: 0,  # building                    -> 0: unlabeled
    51: 0,  # fence                       -> 0: unlabeled
    52: 0,  # other-structure             -> 0: unlabeled
    60: 1,  # lane-marking                -> 1: road
    70: 0,  # vegetation                  -> 0: unlabeled
    71: 0,  # trunk                       -> 0: unlabeled
    72: 0,  # terrain                     -> 0: unlabeled
    80: 0,  # pole                        -> 0: unlabeled
    81: 0,  # traffic-sign                -> 0: unlabeled
    99: 0,  # other-object                -> 0: unlabeled
    252: 3, # moving-car                  -> 3: car
    253: 4, # moving-bicyclist            -> 4: other-vehicle
    254: 4, # moving-person               -> 4: other-vehicle
    255: 4, # moving-motorcyclist         -> 4: other-vehicle
    256: 4, # moving-on-rails             -> 4: other-vehicle
    257: 4, # moving-bus                  -> 4: other-vehicle
    258: 3, # moving-truck                -> 3: car
    259: 4  # moving-other-vehicle        -> 4: other-vehicle
}
metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=259)
input_modality = dict(use_lidar=True, use_camera=False)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/semantickitti/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

pre_transform = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1])
]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.1415926, 3.1415926],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1]),
    dict(
        type='FrustumMix',
        H=32,
        W=1024,
        fov_up=23.0,
        fov_down=-23.0,
        num_areas=[3, 4, 5, 6],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='InstanceCopy',
        instance_classes=[1, 2, 3, 4],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='RangeInterpolation',
        H=32,
        W=4069,
        fov_up=23.0,
        fov_down=-23.0,
        ignore_index=0),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RangeInterpolation',
        H=32,
        W=4096,
        fov_up=23.0,
        fov_down=-23.0,
        ignore_index=0),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=['num_points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RangeInterpolation',
        H=32,
        W=4096,
        fov_up=23.0,
        fov_down=-23.0,
        ignore_index=0),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[-3.1415926, 3.1415926],
                            scale_ratio_range=[0.95, 1.05],
                            translation_std=[0.1, 0.1, 0.1])
                    ],
                    [
                        dict(
                            type='Pack3DDetInputs',
                            keys=['points'],
                            meta_keys=['num_points'])
                    ]])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='MLDAS_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=0,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='MLDAS_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=0,
        test_mode=True,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel')
