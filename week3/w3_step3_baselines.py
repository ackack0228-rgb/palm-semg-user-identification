# 3주차 실습 · 단계 3 : 베이스라인과 비교
#
# 2주차와 "같은 데이터 분할(step6_dataset.npz, 시행 단위 8:2, seed=42)"로 베이스라인
# 모델을 학습한다. 학습 조건은 common.train_model()로 DenseNet161(2주차 step13 seed1)과
# 완전히 동일하게 맞췄다.
# DenseNet161은 2주차에 이미 같은 레시피로 학습해둔 week2/step13_model_seed1.pt를
# 그대로 쓰므로 여기서 다시 학습하지 않는다(학습 시간 1400.1초, week2 로그 참고).
#
# 사용법: ../.venv/Scripts/python -u w3_step3_baselines.py <name> [seed]
#   name: cnn2d | resnet18 (3주차 슬라이드 단계 3의 베이스라인 A·B)
#         cnn1d | cnn_bilstm | cnn_lstm_attn | convnext | tri_ccnn | e2cnn
#         (강의 PDF 37쪽·논문 Table 3의 나머지 베이스라인 6종 — 논문 8종 전부 비교)
# seed 생략 시 1. 슬라이드 순위 규칙("3회 실행 평균")을 위해 seed 2, 3도 학습한다.
# 결과: models/<name>.pt, results/train_<name>.json (seed 1)
#       models/<name>_seed<N>.pt, results/train_<name>_seed<N>.json (seed 2, 3)

import os
import sys
import json
import numpy as np
import torch
from common import BASE_DIR, set_seed, build_model, count_params, train_model, load_split

NAME = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SUFFIX = '' if SEED == 1 else f'_seed{SEED}'

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

    Xtr, ytr, _, _ = load_split(NAME)   # cnn1d만 원 신호 (2, 300), 나머지는 CWT (3, 32, 300)
    print(NAME, 'train:', Xtr.shape, flush=True)

    set_seed(SEED)
    model = build_model(NAME)
    n_params = count_params(model)
    model, train_sec, log = train_model(model, Xtr, ytr, seed=SEED, tag=f'[{NAME} seed{SEED}]')

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'models', f'{NAME}{SUFFIX}.pt'))
    with open(os.path.join(BASE_DIR, 'results', f'train_{NAME}{SUFFIX}.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': NAME, 'seed': SEED, 'params': n_params,
                   'train_seconds': round(train_sec, 1), 'log': log}, f, indent=2)
    print(f'[{NAME}] done  params={n_params:,}  train={train_sec:.1f}s', flush=True)
