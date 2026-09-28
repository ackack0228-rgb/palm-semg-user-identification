# 2주차 실습 · 단계 15 (추가, 진단): 5개 모델 앙상블의 5x5 혼동행렬
#
# step14에서 92.11%까지 올랐지만 E 피험자만 계속 80% 안팎에 머문다. D처럼 모델마다
# 정확도가 크게 요동치는(variance) 패턴이 아니라 어느 모델/앙상블에서도 꾸준히 낮게
# 나오는 걸 보면, E는 "운 나쁘게 틀리는 것"이 아니라 특정 다른 피험자와 신호 패턴이
# 비슷해서 구조적으로 헷갈리는 것일 가능성이 크다. 어느 클래스로 잘못 예측되는지
# 5x5 혼동행렬로 직접 확인한다.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

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

    models = []
    for ckpt in CHECKPOINTS:
        m = build_model(dropout='step13' in ckpt).to(dev)
        m.load_state_dict(torch.load(os.path.join(BASE_DIR, ckpt), map_location=dev))
        m.eval()
        models.append(m)

    cm = np.zeros((5, 5), dtype=int)  # rows = true, cols = pred

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            probs_sum = None
            for m in models:
                probs = torch.softmax(m(xb), dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            pred = probs_sum.argmax(dim=1)
            for t, p in zip(yb.tolist(), pred.tolist()):
                cm[t, p] += 1

    lines = ['confusion matrix (rows=true, cols=pred):']
    header = '      ' + '  '.join(f'{s:>4}' for s in SUBJECTS)
    lines.append(header)
    for i, s in enumerate(SUBJECTS):
        row = f'true {s}: ' + '  '.join(f'{cm[i, j]:>4}' for j in range(5))
        lines.append(row)

    lines.append('')
    lines.append('per-subject misclassification breakdown:')
    for i, s in enumerate(SUBJECTS):
        total = cm[i].sum()
        correct = cm[i, i]
        errs = [(SUBJECTS[j], cm[i, j]) for j in range(5) if j != i and cm[i, j] > 0]
        errs.sort(key=lambda x: -x[1])
        err_str = ', '.join(f'{name}={cnt}' for name, cnt in errs) if errs else 'none'
        lines.append(f'  {s}: correct={correct}/{total}  misclassified_as: {err_str}')

    print('\n'.join(lines))

    out_path = os.path.join(BASE_DIR, 'step15_confusion_matrix.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved:', out_path)
