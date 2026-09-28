# 3주차 실습 · 단계 3 : 베이스라인과 비교
#
# 2주차와 "같은 데이터 분할(step6_dataset.npz, 시행 단위 8:2, seed=42)"로 베이스라인
# 모델을 학습한다. 학습 조건은 common.train_model()로 DenseNet161(2주차 step13 seed1)과
# 완전히 동일하게 맞췄다.
# DenseNet161은 2주차에 이미 같은 레시피로 학습해둔 week2/step13_model_seed1.pt를
# 그대로 쓰므로 여기서 다시 학습하지 않는다(학습 시간 1400.1초, week2 로그 참고).
#
# 사용법: ../.venv/Scripts/python -u w3_step3_baselines.py cnn2d
#         ../.venv/Scripts/python -u w3_step3_baselines.py resnet18
# 결과: models/<name>.pt, results/train_<name>.json (학습 시간·파라미터 수·loss 로그)

import os
import sys
import json
import numpy as np
import torch
from common import BASE_DIR, WEEK2_DIR, set_seed, build_model, count_params, train_model

NAME = sys.argv[1]
SEED = 1

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

    d = np.load(os.path.join(WEEK2_DIR, 'step6_dataset.npz'))
    Xtr, ytr = d['Xtr'], d['ytr']
    print(NAME, 'train:', Xtr.shape, flush=True)

    set_seed(SEED)
    model = build_model(NAME)
    n_params = count_params(model)
    model, train_sec, log = train_model(model, Xtr, ytr, seed=SEED, tag=f'[{NAME}]')

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'models', f'{NAME}.pt'))
    with open(os.path.join(BASE_DIR, 'results', f'train_{NAME}.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': NAME, 'seed': SEED, 'params': n_params,
                   'train_seconds': round(train_sec, 1), 'log': log}, f, indent=2)
    print(f'[{NAME}] done  params={n_params:,}  train={train_sec:.1f}s', flush=True)
