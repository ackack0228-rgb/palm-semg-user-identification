# 2주차 실습 · 단계 20 (추가): 스케일32 5개 + 스케일64 1개, 총 6개 모델 앙상블
#
# step17(스케일64 단일 모델, 90.84%)의 혼동행렬(step19)을 보면 E->B 비중이 71%->56%로
# 줄고 D/C/A로 오류가 분산됐다 — 스케일32 모델들과 "다르게" 틀린다는 뜻이므로, 서로
# 다른 특징 표현(scale 32 vs 64)을 앙상블하면 각자의 편향을 상쇄할 수 있을 거라는
# 가설을 검증한다.
#
# step6_dataset.npz(스케일32)와 step16_dataset_scale64.npz(스케일64)의 테스트셋 순서가
# 동일함을 이미 확인했으므로(같은 파일 목록, 같은 seed=42 분할, 같은 순서로 윈도우
# 생성), 같은 배치 인덱스의 yte가 서로 대응된다. 모델마다 자기 스케일에 맞는 입력을
# 넣고 softmax를 함께 평균낸다.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = ['A', 'B', 'C', 'D', 'E']
SCALE32_CHECKPOINTS = [
    'step10_model_45epoch.pt',
    'step11_model_45epoch.pt',
    'step13_model_seed1.pt',
    'step13_model_seed2.pt',
    'step13_model_seed3.pt',
]
SCALE64_CHECKPOINT = 'step17_model_scale64.pt'


def build_model(dropout=False):
    model = densenet161(weights=None)
    if dropout:
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
    else:
        model.classifier = nn.Linear(2208, 5)
    return model


if __name__ == '__main__':
    d32 = np.load(os.path.join(BASE_DIR, 'step6_dataset.npz'))
    d64 = np.load(os.path.join(BASE_DIR, 'step16_dataset_scale64.npz'))
    Xte32 = torch.from_numpy(d32['Xte'])
    Xte64 = torch.from_numpy(d64['Xte'])
    yte = torch.from_numpy(d32['yte'])
    assert np.array_equal(d32['yte'], d64['yte']), 'test set label order mismatch between scale32/scale64'

    loader32 = DataLoader(TensorDataset(Xte32, yte), batch_size=32, shuffle=False)
    loader64 = DataLoader(TensorDataset(Xte64, yte), batch_size=32, shuffle=False)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    models32 = []
    for ckpt in SCALE32_CHECKPOINTS:
        m = build_model(dropout='step13' in ckpt).to(dev)
        m.load_state_dict(torch.load(os.path.join(BASE_DIR, ckpt), map_location=dev))
        m.eval()
        models32.append(m)

    model64 = build_model(dropout=True).to(dev)
    model64.load_state_dict(torch.load(os.path.join(BASE_DIR, SCALE64_CHECKPOINT), map_location=dev))
    model64.eval()

    correct_total = 0
    n_total = 0
    correct_per_class = Counter()
    total_per_class = Counter()

    with torch.no_grad():
        for (xb32, yb32), (xb64, yb64) in zip(loader32, loader64):
            assert torch.equal(yb32, yb64)
            xb32, xb64, yb = xb32.to(dev), xb64.to(dev), yb32.to(dev)
            probs_sum = None
            for m in models32:
                probs = torch.softmax(m(xb32), dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs64 = torch.softmax(model64(xb64), dim=1)
            probs_sum = probs_sum + probs64
            pred = probs_sum.argmax(dim=1)
            correct = (pred == yb)
            correct_total += correct.sum().item()
            n_total += len(yb)
            for label, is_correct in zip(yb.tolist(), correct.tolist()):
                total_per_class[label] += 1
                correct_per_class[label] += int(is_correct)

    acc = correct_total / n_total * 100
    print(f'ensemble(6 models: 5xscale32 + 1xscale64) test accuracy: {acc:.2f}%  ({correct_total}/{n_total})')

    lines = [
        'checkpoints: ' + ', '.join(SCALE32_CHECKPOINTS + [SCALE64_CHECKPOINT + ' (scale64)']),
        f'test_accuracy: {acc:.2f}% ({correct_total}/{n_total})',
    ]
    for i, s in enumerate(SUBJECTS):
        c, t = correct_per_class[i], total_per_class[i]
        line = f'{s}: {c}/{t} ({c/t*100:.1f}%)'
        print(' ', line)
        lines.append(line)

    out_path = os.path.join(BASE_DIR, 'step20_eval_ensemble6.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', out_path)
