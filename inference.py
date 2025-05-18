import os
import argparse
import numpy as np
import torch
import mmengine
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.apis import init_model


def parse_args():
    parser = argparse.ArgumentParser(description='FRNet 추론 결과 저장')
    parser.add_argument('config', help='테스트 설정 파일 경로')
    parser.add_argument('checkpoint', help='체크포인트 파일')
    parser.add_argument('--data-root', default='data/semantickitti/', help='데이터 루트 경로')
    parser.add_argument('--output-dir', default='output/', help='출력 디렉토리')
    parser.add_argument('--sequences', nargs='+', default=['08'], help='처리할 시퀀스 목록')
    args = parser.parse_args()
    return args


def create_new_mapping():
    # """FRNet 모델의 출력 클래스(0-19)를 새로운 5개 클래스로 매핑합니다."""
    # # 기본적으로 모든 클래스를 4(unlabeled)로 설정
    # frnet_to_new = {i: 4 for i in range(20)}
    
    # # 현재 클래스 이름:
    # # car               -> car
    # # bicycle           -> other-vehicle
    # # motorcycle        -> other-vehicle
    # # truck             -> car
    # # bus               -> other-vehicle
    # # preson            -> other-vehicle
    # # bicyclist         -> other-vehicle
    # # motorcyclist      -> other-vehicle
    # # road              -> road
    # # parking           -> road
    # # sidewalk          -> sidewalk
    # # other-ground      -> unlabeled
    # # building          -> unlabeled
    # # fence             -> unlabeled
    # # vegtation         -> unlabeled
    # # trunck            -> unlabeled
    # # terrian           -> unlabeled
    # # pole              -> unlabeled
    # # traffic-sign      -> unlabeled

    
    # # 자동차(car) 클래스로 매핑 - 0
    # frnet_to_new[0] = 0  # car
    # frnet_to_new[3] = 0  # truck
    
    # # 기타 차량(other-vehicle) 클래스로 매핑 - 1
    # frnet_to_new[1] = 1  # bicycle
    # frnet_to_new[2] = 1  # motorcycle
    # frnet_to_new[4] = 1  # bus 
    # frnet_to_new[5] = 1  # preson 
    # frnet_to_new[6] = 1  # bicyclist
    # frnet_to_new[7] = 1  # motorcyclist
    
    # # 도로(road) 클래스로 매핑 - 2
    # frnet_to_new[8] = 2  # road
    # frnet_to_new[9] = 2  # parking
    
    # # 인도(sidewalk) 클래스로 매핑 - 3
    # frnet_to_new[10] = 3  # sidewalk

    """FRNet 모델의 출력 클래스(0-16)를 새로운 5개 클래스로 매핑합니다."""
    # 기본적으로 모든 클래스를 4(unlabeled)로 설정
    frnet_to_new = {i: 4 for i in range(17)}
    
    # 현재 클래스 이름:
    # 차단물            -> unlabeled
    # 자전거            -> other-vehicle
    # 버스              -> car
    # 자동차            -> car
    # 건설 장비         -> other-vehicle
    # 오토바이          -> other-vehicle
    # 보행자            -> other-vehicle
    # 교통 콘           -> unlabeled
    # 트레일러          -> other-vehicle
    # 트럭              -> car
    # 주행 가능 표면    -> road
    # 기타 평면         -> road
    # 인도              -> sidewalk
    # 지형              -> unlabeled
    # 인공 구조물       -> unlabeled
    # 식생              -> unlabeled

    
    # 자동차(car) 클래스로 매핑 - 0
    frnet_to_new[2] = 0  # 버스
    frnet_to_new[3] = 0  # 자동차
    frnet_to_new[9] = 0  # 트럭
    
    # 기타 차량(other-vehicle) 클래스로 매핑 - 1
    frnet_to_new[1] = 1  # 자전거
    frnet_to_new[4] = 1  # bus 
    frnet_to_new[5] = 1  # preson 
    frnet_to_new[6] = 1  # bicyclist
    frnet_to_new[8] = 1  # motorcyclist
    
    # 도로(road) 클래스로 매핑 - 2
    frnet_to_new[10] = 2  # 주행 가능 표면
    frnet_to_new[11] = 2  # 기타 평면
    
    # 인도(sidewalk) 클래스로 매핑 - 3
    frnet_to_new[10] = 3  # 인도
    
    return frnet_to_new


def apply_new_mapping(original_labels, mapping):
    """원본 레이블에 새 매핑을 적용합니다."""
    new_labels = np.copy(original_labels)
    
    # 고유한 레이블 값에 대해 매핑 적용
    unique_labels = np.unique(original_labels)
    for label in unique_labels:
        mask = (original_labels == label)
        new_labels[mask] = mapping.get(label, 4)  # 기본값은 4(unlabeled)
    
    return new_labels


def main():
    args = parse_args()
    
    # 모델 초기화
    model = init_model(args.config, args.checkpoint, device='cuda:0')
    # 디버깅용
    # model = init_model(args.config, args.checkpoint, device='cpu')
    
    # 새로운 클래스 매핑 생성
    new_mapping = create_new_mapping()
    
    # 출력 디렉토리 생성
    for seq in args.sequences:
        output_path = os.path.join(args.output_dir, 'sequences', seq, 'predictions')
        os.makedirs(output_path, exist_ok=True)
        print(f"출력 디렉토리 생성됨: {output_path}")
    
    # 각 시퀀스에 대해 처리
    for seq in args.sequences:
        seq_dir = os.path.join(args.data_root, 'sequences', seq, 'velodyne')
        print(f"시퀀스 {seq} 처리 중...")
        
        file_list = sorted([f for f in os.listdir(seq_dir) if f.endswith('.bin')])
        
        for file_name in mmengine.track_iter_progress(file_list):
            file_path = os.path.join(seq_dir, file_name)
            
            # 포인트 클라우드 데이터 로드
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            
            # (0,0,0) 포인트 필터링 - x, y, z 좌표가 모두 0인 포인트 제외
            non_zero_mask = ~np.all(points[:, :3] == 0, axis=1)
            points = points[non_zero_mask]
            
            # 원본 포인트 수 저장 (결과에 필요할 수 있음)
            original_num_points = points.shape[0]
            
            # 필터링 결과 출력 (디버깅용)
            # print(f"파일: {file_name}, 총 포인트: {original_num_points}, 0,0,0 포인트 필터링 후: {points.shape[0]}")
            
            # intensity 정규화 (필요한 경우)
            if points[:, 3].max() > 1.0:  # 범위가 [0,255] 같은 경우
                points[:, 3] = points[:, 3] / 65535.0
            
            points_tensor = torch.from_numpy(points).float().cuda()
            
            # Det3DDataSample 생성 및 필요한 속성 설정
            data_sample = Det3DDataSample(metainfo=dict(num_points=points.shape[0]))
            
            # PointData 객체로 gt_pts_seg 설정 (빈 마스크)
            data_sample.gt_pts_seg = PointData()
            data_sample.gt_pts_seg.pts_semantic_mask = torch.zeros(points.shape[0], dtype=torch.long)
            
            # 수동으로 데이터 전처리 단계 수행
            with torch.no_grad():
                # 모델 입력 형식으로 데이터 구성
                model_inputs = dict(
                    inputs=dict(points=[points_tensor]),
                    data_samples=[data_sample]
                )
                
                # 데이터 전처리 수행
                processed_data = model.data_preprocessor(model_inputs, False)
                
                # 모델 추론
                results = model.forward(**processed_data, mode='predict')
                
                # 원래 20개 클래스로 예측된 레이블
                original_pred_labels = results[0].pred_pts_seg.pts_semantic_mask.cpu().numpy()
                
                # 새로운 5개 클래스로 매핑 적용
                new_pred_labels = apply_new_mapping(original_pred_labels, new_mapping)
            
            # 원본 데이터와 새로 매핑된 예측 레이블 결합
            final_result = np.hstack((points, new_pred_labels.reshape(-1, 1)))
            
            # 결과 저장
            output_path = os.path.join(args.output_dir, 'sequences', seq, 'predictions', file_name)
            final_result.astype(np.float32).tofile(output_path)
    
    print("모든 처리 완료!")


if __name__ == '__main__':
    main()