import torch

# 1. 체크포인트 로드 (CPU로)
ckpt = torch.load('checkpoints/frnet-semantickitti_seg.pth', map_location='cpu')

# 2. 최상위 키 확인
print("Top-level keys:", ckpt.keys())
# → 보통 'state_dict', 'meta' 등이 나옵니다.

# 3. meta 안에 config가 있는지 확인
if 'meta' in ckpt:
    meta = ckpt['meta']
    print("Meta keys:", meta.keys())
    if 'config' in meta:
        config_text = meta['config']
        print("Config (일부):\n", config_text.splitlines()[:20])
    else:
        print("‘config’ 항목이 meta에 없습니다.")
else:
    print("‘meta’ 항목이 없습니다. 이 체크포인트는 config를 포함하지 않습니다.")
