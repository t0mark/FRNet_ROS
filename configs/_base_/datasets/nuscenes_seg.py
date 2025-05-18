# For NuScenes we usually do 16-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'NuScenesSegDataset'
data_root = 'data/nuscenes/'
class_names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]
'''
class_names = [
    차단물, 자전거, 버스, 자동차, 건설 장비, 오토바이, 보행자, 교통 콘, 트레일러,
    트럭, 주행 가능 표면, 기타 평면, 인도, 지형, 인공 구조물, 식생
]
'''
labels_map = {
    0: 16,  # 잡음 -> unlabeled
    1: 16,  # 동물 -> unlabeled
    2: 6,   # 보행자(성인) -> 보행자
    3: 6,   # 보행자(어린이) -> 보행자
    4: 6,   # 보행자(건설근로자) -> 보행자
    5: 16,  # 보행자(이동 수단 탑승자) -> unlabeled
    6: 6,   # 보행자(경찰관) -> 보행자
    7: 16,  # 보행자(유모차  탑승자) -> unlabeled
    8: 16,  # 보행자(휠체어 탑승자) -> unlabeled
    9: 0,   # 이동 물체(벽) -> 차단물
    10: 16, # 이동 물체(잔해) -> unlabeled
    11: 16, # 이동 물체(밀거나 당길 수 있는 물체) -> unlabeled
    12: 7,  # 이동 물체(트래픽 콘) -> 교통 콘
    13: 16, # 고정 물체(자전거 거치대) -> unlabeled
    14: 1,  # 차량(자전거) -> 자전거
    15: 2,  # 차량(굴절 버스) -> 버스
    16: 2,  # 차량(일반 버스) -> 버스
    17: 3,  # 차량(자동차) -> 자동차
    18: 4,  # 차량(건설 장비) -> 건설 장비
    19: 16, # 차량(구급차) -> unlabeled
    20: 16, # 차량(경찰차) -> unlabeled
    21: 5,  # 차량(오토바이) -> 오토바이
    22: 8,  # 차량(견인차) -> 트레일러
    23: 9,  # 차량(트럭) -> 트럭
    24: 10, # 지면(주행 가능 표면) -> 주행 가능 표면
    25: 11, # 지면(기타) -> 기타 평면
    26: 12, # 지면(보도) -> 인도
    27: 13, # 지면(흙) -> 지형
    28: 14, # 고정 물체(인공 구조물) -> 인공 구조물
    29: 16, # 고정 물체(기타) -> unlabeled
    30: 15, # 고정 물체(식생) -> 식생
    31: 16  # 차량(자차) -> unlabeled
}
metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=31)
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    img='',
    pts_semantic_mask='lidarseg/v1.0-trainval')

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/nuscenes/'

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
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
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
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
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
        W=480,
        fov_up=10.0,
        fov_down=-30.0,
        num_areas=[3, 4, 5, 6],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='InstanceCopy',
        instance_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        pre_transform=pre_transform,
        prob=1.0),
    dict(
        type='RangeInterpolation',
        H=32,
        W=1920,
        fov_up=10.0,
        fov_down=-30.0,
        ignore_index=16),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RangeInterpolation',
        H=32,
        W=1920,
        fov_up=10.0,
        fov_down=-30.0,
        ignore_index=16),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=['num_points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RangeInterpolation',
        H=32,
        W=1920,
        fov_up=10.0,
        fov_down=-30.0,
        ignore_index=16),
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
        ann_file='nuscenes_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16,
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
        ann_file='nuscenes_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16,
        test_mode=True,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel')
