# 2주차 실습 · 단계 14 (추가): 5개 모델(step10, step11, step13 seed1/2/3) 전체 앙상블 평가
#
# step12에서 두 모델(step10+step11) 앙상블만으로 91.58%까지 오르는 걸 확인했다.
# 이번엔 Dropout을 추가해 새로 학습한 시드 1/2/3(step13)까지 합쳐 총 5개 모델을
# softmax 평균 앙상블로 평가한다. step12_diagnose_ensemble.py를 복사해 CHECKPOINTS만
# 5개로 늘렸다.
#
# 콘솔 출력이 cp949로 한글이 깨지는 문제가 반복돼서(위 CLAUDE.md "작업 중 겪은 이슈"
# 참고), 이 스크립트는 step8_evaluate.py처럼 결과를 영문 라벨로 파일에 직접 저장한다.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = ['A', 'B', 'C', 'D', 'E']
CHECKPOINTS = [
    'step10_model_45epoch.pt',
    'step11_model_45epoch.pt',
    'step13_model_seed1.pt',
    'step13_model_seed2.pt',
    'step13_model_seed3.pt',
]


def build_model(dropout=False):
    model = densenet161(weights=None)
    if dropout:
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
    else:
        model.classifier = nn.Linear(2208, 5)
    return model


if __name__ == '__main__':
    d = np.load(os.path.join(BASE_DIR, 'step6_dataset.npz'))
    Xte = torch.from_numpy(d['Xte'])
    yte = torch.from_numpy(d['yte'])
    loader = DataLoader(TensorDataset(Xte, yte), batch_size=32)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    models = []
    for ckpt in CHECKPOINTS:
        m = build_model(dropout='step13' in ckpt).to(dev)
        m.load_state_dict(torch.load(os.path.join(BASE_DIR, ckpt), map_location=dev))
        m.eval()
        models.append(m)

    correct_total = 0
    n_total = 0
    correct_per_class = Counter()
    total_per_class = Counter()

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            probs_sum = None
            for m in models:
                logits = m(xb)
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            pred = probs_sum.argmax(dim=1)
            correct = (pred == yb)
            correct_total += correct.sum().item()
            n_total += len(yb)
            for label, is_correct in zip(yb.tolist(), correct.tolist()):
                total_per_class[label] += 1
                correct_per_class[label] += int(is_correct)

    acc = correct_total / n_total * 100
    print(f'ensemble test accuracy: {acc:.2f}%  ({correct_total}/{n_total})')

    lines = [
        'checkpoints: ' + ', '.join(CHECKPOINTS),
        f'test_accuracy: {acc:.2f}% ({correct_total}/{n_total})',
    ]
    for i, s in enumerate(SUBJECTS):
        c, t = correct_per_class[i], total_per_class[i]
        line = f'{s}: {c}/{t} ({c/t*100:.1f}%)'
        print(' ', line)
        lines.append(line)

    out_path = os.path.join(BASE_DIR, 'step14_eval_ensemble5.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', out_path)
