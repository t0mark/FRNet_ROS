from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import BaseDataPreprocessor
from torch import Tensor


@MODELS.register_module()
class FrustumRangePreprocessor(BaseDataPreprocessor):
    """Frustum-Range Segmentor pre-processor for frustum region group.

    Args:
        H (int): Height of the 2D representation.
        W (int): Width of the 2D representation.
        fov_up (float): Front-of-View at upward direction of the sensor.
        fov_down (float): Front-of-View at downward direction of the sensor.
        ignore_index (int): The label index to be ignored.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
    """

    def __init__(self,
                 H: int,
                 W: int,
                 fov_up: float,
                 fov_down: float,
                 ignore_index: int,
                 non_blocking: bool = False) -> None:
        super(FrustumRangePreprocessor,
              self).__init__(non_blocking=non_blocking)
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180 * np.pi
        self.fov_down = fov_down / 180 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.ignore_index = ignore_index

        # 디버깅용
        # print(f"[CONFIG] H={self.H}, W={self.W}, fov_up={self.fov_up}, fov_down={self.fov_down}")

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform frustum region group based on ``BaseDataPreprocessor``.

        Args:
            data (dict): Data from dataloader. The dict contains the whole
                batch data.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        data.setdefault('data_samples', None)

        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        assert 'points' in inputs
        batch_inputs['points'] = inputs['points']

        voxel_dict = self.frustum_region_group(inputs['points'], data_samples)
        batch_inputs['voxels'] = voxel_dict

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    @torch.no_grad()
    def frustum_region_group(self, points: List[Tensor], data_samples: SampleList) -> dict:
        voxel_dict = dict()
        coors = []
        voxels = []

        # 디버깅 정보 저장용 리스트
        # debug_info = {}

        for i, res in enumerate(points):
            # 디버깅 정보 초기화
            # batch_debug = {}
            
            depth = torch.linalg.norm(res[:, :3], 2, dim=1)
            yaw = -torch.atan2(res[:, 1], res[:, 0])
            pitch = torch.arcsin(res[:, 2] / depth)

            # 디버깅 - 중간값 통계 저장
            # batch_debug['depth'] = {
            #     'min': float(torch.min(depth)), 'max': float(torch.max(depth)), 
            #     'mean': float(torch.mean(depth)), 'nan': int(torch.isnan(depth).sum()),
            #     'inf': int(torch.isinf(depth).sum())
            # }
            # batch_debug['yaw'] = {
            #     'min': float(torch.min(yaw)), 'max': float(torch.max(yaw)), 
            #     'mean': float(torch.mean(yaw)), 'nan': int(torch.isnan(yaw).sum()),
            #     'inf': int(torch.isinf(yaw).sum())
            # }
            # batch_debug['pitch'] = {
            #     'min': float(torch.min(pitch)), 'max': float(torch.max(pitch)), 
            #     'mean': float(torch.mean(pitch)), 'nan': int(torch.isnan(pitch).sum()),
            #     'inf': int(torch.isinf(pitch).sum())
            # }

            coors_x = 0.5 * (yaw / np.pi + 1.0)
            coors_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

            # 디버깅 - 정규화 좌표 통계 저장
            # batch_debug['norm_x'] = {
            #     'min': float(torch.min(coors_x)), 'max': float(torch.max(coors_x)), 
            #     'mean': float(torch.mean(coors_x)), 'nan': int(torch.isnan(coors_x).sum()),
            #     'inf': int(torch.isinf(coors_x).sum()),
            #     'out_of_range': int(((coors_x < 0) | (coors_x > 1)).sum())
            # }
            # batch_debug['norm_y'] = {
            #     'min': float(torch.min(coors_y)), 'max': float(torch.max(coors_y)), 
            #     'mean': float(torch.mean(coors_y)), 'nan': int(torch.isnan(coors_y).sum()),
            #     'inf': int(torch.isinf(coors_y).sum()),
            #     'out_of_range': int(((coors_y < 0) | (coors_y > 1)).sum())
            # }

            # scale to image size using angular resolution
            coors_x *= self.W
            coors_y *= self.H

            # 스케일링 전 좌표 저장 (디버깅용)
            pre_clamped_x = coors_x.clone()
            pre_clamped_y = coors_y.clone()

            # round and clamp for use as index
            coors_x = torch.floor(coors_x)
            coors_x = torch.clamp(coors_x, min=0, max=self.W - 1).type(torch.int64)

            coors_y = torch.floor(coors_y)
            coors_y = torch.clamp(coors_y, min=0, max=self.H - 1).type(torch.int64)

            # 디버깅 - 최종 픽셀 좌표 통계 저장
            # batch_debug['pixel_x'] = {
            #     'min': float(torch.min(pre_clamped_x)), 'max': float(torch.max(pre_clamped_x)), 
            #     'mean': float(torch.mean(pre_clamped_x)), 
            #     'out_of_range': int(((pre_clamped_x < 0) | (pre_clamped_x >= self.W)).sum()),
            #     'clamped_min': int((pre_clamped_x < 0).sum()), 
            #     'clamped_max': int((pre_clamped_x >= self.W).sum())
            # }
            # batch_debug['pixel_y'] = {
            #     'min': float(torch.min(pre_clamped_y)), 'max': float(torch.max(pre_clamped_y)), 
            #     'mean': float(torch.mean(pre_clamped_y)), 
            #     'out_of_range': int(((pre_clamped_y < 0) | (pre_clamped_y >= self.H)).sum()),
            #     'clamped_min': int((pre_clamped_y < 0).sum()), 
            #     'clamped_max': int((pre_clamped_y >= self.H).sum())
            # }
            
            # FOV 정보 추가
            # batch_debug['fov_settings'] = {
            #     'fov_up': float(self.fov_up), 
            #     'fov_down': float(self.fov_down), 
            #     'total_fov': float(self.fov),
            #     'H': self.H, 
            #     'W': self.W
            # }
            
            # 포인트 수 정보
            # batch_debug['num_points'] = len(res)
            
            # 디버깅 정보 저장
            # debug_info[f'batch_{i}'] = batch_debug
            
            res_coors = torch.stack([coors_y, coors_x], dim=1)
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            coors.append(res_coors)
            voxels.append(res)

            # 추가 디버깅 - 샘플 포인트 출력
            # sample_indices = torch.randint(0, len(res), (min(5, len(res)),))
            # print(f"\n===== 배치 {i} 샘플 포인트 =====")
            # for idx in sample_indices:
            #     idx = idx.item()
            #     print(f"Point[{idx}]: 원본좌표=({res[idx, 0]:.2f}, {res[idx, 1]:.2f}, {res[idx, 2]:.2f}), "
            #         f"depth={depth[idx]:.2f}, yaw={yaw[idx]:.2f}, pitch={pitch[idx]:.2f}, "
            #         f"정규화좌표=({coors_x[idx]:.2f}, {coors_y[idx]:.2f})")

        # 전체 디버깅 정보 출력
        # import json
        # print("\n===== Frustum 투영 디버깅 정보 =====")
        # print(json.dumps(debug_info, indent=2))
        
        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict
