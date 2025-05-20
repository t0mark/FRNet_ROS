import argparse
from os import path as osp
from pathlib import Path

import mmengine

total_num = {
    0: 5195,
    1: 600
}
fold_split = {
    'train': [0],          # 0번 시퀀스로 훈련 (5195개)
    'val': [1],            # 1번 시퀀스로 검증 (600개)
    'trainval': [0, 1],    # 전체 데이터 (5795개)
    'test': [],            # 테스트는 비워두기
}
split_list = ['train', 'val', 'trainval', 'test']  # 'valid'를 'val'로 변경


def get_MLDAS_info(split: str) -> dict:
    data_infos = dict()
    data_infos['metainfo'] = dict(dataset='MLDAS')
    data_list = []
    for i_folder in fold_split[split]:
        for j in range(total_num[i_folder]):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                    osp.join('sequences',
                             str(i_folder).zfill(2), 'ouster',
                             str(j).zfill(6) + '.bin'),
                    'num_pts_feats':
                    4
                },
                'pts_semantic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         str(j).zfill(6) + '.label'),
                'sample_idx':
                str(i_folder).zfill(2) + str(j).zfill(6)
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_MLDAS_info_file(pkl_prefix: str, save_path: str) -> None:
    print('Generate info.')
    save_path = Path(save_path)

    MLDAS_infos_train = get_MLDAS_info(split='train')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'MLDAS info train file is saved to {filename}')
    mmengine.dump(MLDAS_infos_train, filename)

    MLDAS_infos_val = get_MLDAS_info(split='val')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'MLDAS info val file is saved to {filename}')
    mmengine.dump(MLDAS_infos_val, filename)

    MLDAS_infos_trainval = get_MLDAS_info(split='trainval')
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'MLDAS info trainval file is saved to {filename}')
    mmengine.dump(MLDAS_infos_trainval, filename)

    MLDAS_infos_test = get_MLDAS_info(split='test')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'MLDAS info test file is saved to {filename}')
    mmengine.dump(MLDAS_infos_test, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/MLDAS',  # 기본 출력 경로를 MLDAS로 변경
        required=False,
        help='output path of pkl')
    parser.add_argument('--extra-tag', type=str, default='MLDAS')  # 기본값을 MLDAS로 변경
    args = parser.parse_args()
    create_MLDAS_info_file(args.extra_tag, args.out_dir)