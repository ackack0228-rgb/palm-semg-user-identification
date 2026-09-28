# 2주차 실습 · 단계 12 (추가, 진단): 왜 96%에 못 미치는지 확인 + 앙상블 실험
#
# 관찰된 현상:
#   step10(웜스타트, 증강 없음) 45에폭: 88.21%  (A98 B83 C87 D94 E78)
#   step11(처음부터, 증강+weight decay+cos) 45에폭: 89.05%  (A96 B96 C89 D88 E76)
# 두 모델 다 "전체적으로는" 논문(94%)에 못 미치고, 피험자별 정확도가 두 모델 사이에서
# 크게 요동친다(D: 94%->88%, B: 83%->96%). 테스트 트라이얼이 피험자당 10개뿐이라
# (190 윈도우 / 19 윈도우/트라이얼) 트라이얼 하나가 틀리면 정확도가 10%p씩 흔들린다 —
# 즉 이 변동은 "특정 하이퍼파라미터가 잘못됐다"기보다 작은 테스트셋에서 오는
# 모델별 무작위성(variance)일 가능성이 크다. 이 가설이 맞다면, 여러 모델의 예측을
# 평균 내는 앙상블이 각 모델이 우연히 틀리는 트라이얼을 서로 상쇄해줘서 정확도를
# 올려줄 것이다. 새로 모델을 학습하기 전에, 이미 있는 두 모델(step10, step11)만으로
# 먼저 앙상블 효과가 있는지 빠르게 확인해본다.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = ['A', 'B', 'C', 'D', 'E']
CHECKPOINTS = ['step10_model_45epoch.pt', 'step11_model_45epoch.pt']


def build_model(dropout=False):
    """step13(2026-09-22 Dropout 추가) 이후 체크포인트는 classifier 구조가 다르므로
    dropout=True로 만들어야 state_dict 키가 맞는다."""
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

    print(f'앙상블({" + ".join(CHECKPOINTS)}) 테스트 정확도: {correct_total/n_total*100:.2f}%  ({correct_total}/{n_total})')
    for i, s in enumerate(SUBJECTS):
        c, t = correct_per_class[i], total_per_class[i]
        print(f'  {s}: {c}/{t} ({c/t*100:.1f}%)')
