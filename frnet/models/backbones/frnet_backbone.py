from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor


class BasicBlock(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: Normalization layer after the first convolution layer."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        """nn.Module: Normalization layer after the second convolution layer.
        """
        return getattr(self, self.norm2_name)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@MODELS.register_module()
class FRNetBackbone(BaseModule):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3))
    }

    def __init__(self,
                 in_channels: int,
                 point_in_channels: int,
                 output_shape: Sequence[int],
                 depth: int,
                 stem_channels: int = 128,
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 fuse_channels: Sequence[int] = (256, 128),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 point_norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super(FRNetBackbone, self).__init__(init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for FRNetBackbone.')

        self.block, stage_blocks = self.arch_settings[depth]
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        assert len(stage_blocks) == len(out_channels) == len(strides) == len(
            dilations) == num_stages, \
            'The length of stage_blocks, out_channels, strides and ' \
            'dilations should be equal to num_stages.'
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.point_norm_cfg = point_norm_cfg
        self.act_cfg = act_cfg
        self.stem = self._make_stem_layer(in_channels, stem_channels)
        self.point_stem = self._make_point_layer(point_in_channels,
                                                 stem_channels)
        self.fusion_stem = self._make_fusion_layer(stem_channels * 2,
                                                   stem_channels)

        inplanes = stem_channels
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()
        self.pixel_fusion_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.strides = []
        overall_stride = 1
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.point_fusion_layers.append(
                self._make_point_layer(inplanes + planes, planes))
            self.pixel_fusion_layers.append(
                self._make_fusion_layer(planes * 2, planes))
            self.attention_layers.append(self._make_attention_layer(planes))
            inplanes = planes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        in_channels = stem_channels + sum(out_channels)
        self.fuse_layers = []
        self.point_fuse_layers = []
        for i, fuse_channel in enumerate(fuse_channels):
            fuse_layer = ConvModule(
                in_channels,
                fuse_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            point_fuse_layer = self._make_point_layer(in_channels,
                                                      fuse_channel)
            in_channels = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer)
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

    def _make_stem_layer(self, in_channels: int,
                         out_channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels // 2)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                out_channels // 2,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_point_layer(self, in_channels: int,
                          out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            build_norm_layer(self.point_norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True))

    def _make_fusion_layer(self, in_channels: int,
                           out_channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_attention_layer(self, channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, channels)[1], nn.Sigmoid())

    def make_res_layer(
        self,
        block: nn.Module,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN'),
        act_cfg: ConfigType = dict(type='LeakyReLU')
    ) -> nn.Module:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes)[1])

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        return nn.Sequential(*layers)

    def forward(self, voxel_dict: dict) -> dict:

        point_feats = voxel_dict['point_feats'][-1]
        voxel_feats = voxel_dict['voxel_feats']
        voxel_coors = voxel_dict['voxel_coors']
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1

        x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        x = self.stem(x)
        map_point_feats = self.pixel2point(x, pts_coors, stride=1)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
        point_feats = self.point_stem(fusion_point_feats)
        stride_voxel_coors, frustum_feats = self.point2frustum(
            point_feats, pts_coors, stride=1)
        pixel_feats = self.frustum2pixel(
            frustum_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
        x = self.fusion_stem(fusion_pixel_feats)

        outs = [x]
        out_points = [point_feats]
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # frustum-to-point fusion
            map_point_feats = self.pixel2point(
                x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat((map_point_feats, point_feats),
                                           dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # point-to-frustum fusion
            stride_voxel_coors, frustum_feats = self.point2frustum(
                point_feats, pts_coors, stride=self.strides[i])
            pixel_feats = self.frustum2pixel(
                frustum_feats,
                stride_voxel_coors,
                batch_size,
                stride=self.strides[i])
            fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
            # residual-attentive
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x
            outs.append(x)
            out_points.append(point_feats)

        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(
                    outs[i],
                    size=outs[0].size()[2:],
                    mode='bilinear',
                    align_corners=True)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(self.fuse_layers,
                                                self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        voxel_dict['voxel_feats'] = outs
        voxel_dict['point_feats_backbone'] = out_points
        return voxel_dict

    def frustum2pixel(self,
                      frustum_features: Tensor,
                      coors: Tensor,
                      batch_size: int,
                      stride: int = 1) -> Tensor:
        nx = self.nx // stride
        ny = self.ny // stride
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]),
            dtype=frustum_features.dtype,
            device=frustum_features.device)
        pixel_features[coors[:, 0], coors[:, 1], coors[:,
                                                       2]] = frustum_features
        pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous()
        return pixel_features

    def pixel2point(self,
                    pixel_features: Tensor,
                    coors: Tensor,
                    stride: int = 1) -> Tensor:
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        point_feats = pixel_features[coors[:, 0], coors[:, 1] // stride,
                                     coors[:, 2] // stride]
        return point_feats

    def point2frustum(self, point_features: Tensor, pts_coors: Tensor, stride: int = 1) -> Tuple[Tensor, Tensor]:
        """
        GPU에서 직접 디버깅하는 코드
        """
        print("\n===== point2frustum GPU 디버깅 =====")
        
        # 기본 정보 출력
        print(f"stride: {stride}")
        print(f"self.ny: {self.ny}, self.nx: {self.nx}")
        
        # 텐서 기본 속성 확인 (복사하지 않음)
        print(f"pts_coors 타입: {type(pts_coors)}")
        if isinstance(pts_coors, torch.Tensor):
            print(f"pts_coors shape: {pts_coors.shape}")
            print(f"pts_coors 디바이스: {pts_coors.device}")
            print(f"pts_coors 데이터 타입: {pts_coors.dtype}")
            print(f"pts_coors 연속성: {pts_coors.is_contiguous()}")
        
        # 안전하게 데이터 검사 (값에 직접 접근하지 않음)
        try:
            print("데이터 형태 검사 중...")
            
            # 이미 복사되었기 때문에 clone()은 쓰지 않음
            # 기존 pts_coors 건드리지 않고 새 변수 생성
            print("변환 준비...")
            
            # 메모리 위치만 확인 (값 접근하지 않음)
            print(f"point_features 메모리 주소: {point_features.data_ptr()}")
            print(f"pts_coors 메모리 주소: {pts_coors.data_ptr()}")
            
            # 차원 확인 (값 접근하지 않음)
            print(f"pts_coors 차원 수: {pts_coors.dim()}")
            if pts_coors.dim() != 2:
                print("경고: pts_coors가 2차원 텐서가 아님!")
            
            # 데이터가 유효한지 확인 (값 접근하지 않음)
            if pts_coors.numel() == 0:
                print("경고: pts_coors가 비어 있음!")
                return None, None
                
            # 메모리 배치 확인 - 이 정보는 GPU에서도 안전하게 접근 가능
            if not pts_coors.is_contiguous():
                print("경고: pts_coors가 연속적이지 않음. 연속적으로 만듭니다.")
                # 이 시점에서 에러가 발생하는지 확인
                try:
                    pts_coors = pts_coors.contiguous()
                    print("연속적으로 만들기 성공")
                except Exception as e:
                    print(f"연속적으로 만들기 실패: {e}")
                    return None, None
            
            # 단계별로 진행하며 문제 발생 지점 찾기
            print("단계 1: dims와 요소 수 확인")
            print(f"모든 차원 크기: {[pts_coors.size(i) for i in range(pts_coors.dim())]}")
            
            print("단계 2: 메타데이터 확인")
            print(f"stride 적용 후 유효 높이: {self.ny // stride}")
            print(f"stride 적용 후 유효 너비: {self.nx // stride}")
            
            # 여기까지 잘 진행되면 다음 단계로
            print("단계 3: batch_size 확인을 시도합니다 (첫번째 열)")
            
            # 안전하게 접근하기 위한 방법
            try:
                # 직접 인덱싱하는 대신 안전한 방법으로 첫번째 열 접근
                # 첫번째 열만 선택 (전체 행)
                batch_col = pts_coors.select(1, 0)
                print(f"배치 인덱스 열 접근 성공: shape {batch_col.shape}")
                
                # unique()도 인덱싱 없이 호출할 수 있음
                unique_batches = torch.unique(batch_col)
                print(f"고유 배치 개수: {len(unique_batches)}")
                
                # min, max 값은 안전하게 접근 가능
                batch_min = batch_col.min().item()
                batch_max = batch_col.max().item()
                print(f"배치 인덱스 범위: {batch_min} ~ {batch_max}")
                
                # 각 열에 대해 같은 작업 수행
                print("단계 4: y 좌표 열 확인")
                y_col = pts_coors.select(1, 1)
                y_min = y_col.min().item()
                y_max = y_col.max().item()
                print(f"y 좌표 범위: {y_min} ~ {y_max}")
                
                print("단계 5: x 좌표 열 확인")
                x_col = pts_coors.select(1, 2)
                x_min = x_col.min().item()
                x_max = x_col.max().item()
                print(f"x 좌표 범위: {x_min} ~ {x_max}")
                
                # 범위 밖 값 확인 (안전하게)
                y_max_allowed = self.ny // stride - 1
                x_max_allowed = self.nx // stride - 1
                
                y_out_min = (y_col < 0).sum().item()
                y_out_max = (y_col > y_max_allowed).sum().item()
                print(f"범위 밖 y 좌표: 음수={y_out_min}, 최대초과={y_out_max}")
                
                x_out_min = (x_col < 0).sum().item()
                x_out_max = (x_col > x_max_allowed).sum().item()
                print(f"범위 밖 x 좌표: 음수={x_out_min}, 최대초과={x_out_max}")
                
                # 문제가 있으면 여기서 종료
                if y_out_min > 0 or y_out_max > 0 or x_out_min > 0 or x_out_max > 0:
                    print("⚠️ 범위를 벗어난 좌표가 발견되었습니다!")
                    print("이 부분에서 clamp 처리가 필요합니다.")
                    
                    # 실제 오류가 발생하는 지점으로 진행
                    print("이제 실제 변환을 시도합니다...")
                
            except Exception as e:
                print(f"열 접근 과정에서 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return None, None
                
            # 실제 변환 시도
            print("단계 6: 좌표 변환 시도")
            try:
                # 변환 코드를 한 줄씩 실행하며 확인
                print("6-1: 복제")
                coors = pts_coors.clone()
                print("복제 성공")
                
                print("6-2: y 좌표 변환")
                # 직접 접근 대신 연산 사용
                y_transformed = pts_coors.select(1, 1).div(stride, rounding_mode='floor')
                # (가능하면) 범위 제한
                y_transformed = torch.clamp(y_transformed, 0, y_max_allowed)
                print("y 좌표 변환 성공")
                
                print("6-3: x 좌표 변환")
                x_transformed = pts_coors.select(1, 2).div(stride, rounding_mode='floor')
                # (가능하면) 범위 제한
                x_transformed = torch.clamp(x_transformed, 0, x_max_allowed)
                print("x 좌표 변환 성공")
                
                print("6-4: 새 좌표 할당")
                coors[:, 1] = y_transformed
                coors[:, 2] = x_transformed
                print("새 좌표 할당 성공")
                
                print("변환된 좌표 범위:")
                print(f"y 범위: {coors[:, 1].min().item()} ~ {coors[:, 1].max().item()}")
                print(f"x 범위: {coors[:, 2].min().item()} ~ {coors[:, 2].max().item()}")
                
                # 여기까지 성공했다면 torch.unique 시도
                print("단계 7: torch.unique 시도")
                voxel_coors, inverse_map = torch.unique(coors, return_inverse=True, dim=0)
                print(f"torch.unique 성공! voxel_coors shape: {voxel_coors.shape}")
                
                print("단계 8: scatter_max 시도")
                frustum_features = torch_scatter.scatter_max(point_features, inverse_map, dim=0)[0]
                print(f"scatter_max 성공! frustum_features shape: {frustum_features.shape}")
                
                return voxel_coors, frustum_features
                
            except Exception as e:
                print(f"좌표 변환 과정에서 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return None, None
                
        except Exception as e:
            print(f"디버깅 과정에서 일반 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None, None
