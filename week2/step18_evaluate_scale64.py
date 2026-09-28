# 2주차 실습 · 단계 18 (추가): CWT 스케일64 모델 평가
#
# step8_evaluate.py와 거의 동일하지만, step6_dataset.npz(스케일32) 대신
# step16_dataset_scale64.npz(스케일64)를 읽는다 — 입력 텐서 모양이 (3,32,300)에서
# (3,64,300)으로 바뀌었기 때문에 기존 평가 스크립트를 그대로 쓸 수 없다.
#
# 사용법: ../.venv/Scripts/python step18_evaluate_scale64.py step17_model_scale64.pt

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = ['A', 'B', 'C', 'D', 'E']


def build_model():
    model = densenet161(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
    return model


@torch.no_grad()
def evaluate(model, X, y, dev, batch_size=32):
    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    correct_total = 0
    n_total = 0
    correct_per_class = Counter()
    total_per_class = Counter()
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        pred = model(xb).argmax(dim=1)
        correct = (pred == yb)
        correct_total += correct.sum().item()
        n_total += len(yb)
        for label, is_correct in zip(yb.tolist(), correct.tolist()):
            total_per_class[label] += 1
            correct_per_class[label] += int(is_correct)
    return correct_total / n_total, correct_per_class, total_per_class


if __name__ == '__main__':
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else 'step17_model_scale64.pt'
    ckpt_path = os.path.join(BASE_DIR, ckpt_name)

    d = np.load(os.path.join(BASE_DIR, 'step16_dataset_scale64.npz'))
    Xte = torch.from_numpy(d['Xte'])
    yte = torch.from_numpy(d['yte'])

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model().to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))

    acc, correct_per_class, total_per_class = evaluate(model, Xte, yte, dev)
    print(f'checkpoint: {ckpt_name}')
    print(f'test_accuracy: {acc*100:.2f}%  ({sum(correct_per_class.values())}/{sum(total_per_class.values())})')
    for i, s in enumerate(SUBJECTS):
        c, t = correct_per_class[i], total_per_class[i]
        print(f'  {s}: {c}/{t} ({c/t*100:.1f}%)')

    out_path = os.path.join(BASE_DIR, f'step18_eval_{os.path.splitext(ckpt_name)[0]}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'checkpoint: {ckpt_name}\n')
        f.write(f'test_accuracy: {acc*100:.2f}%\n')
        for i, s in enumerate(SUBJECTS):
            c, t = correct_per_class[i], total_per_class[i]
            f.write(f'{s}: {c}/{t} ({c/t*100:.1f}%)\n')
    print('saved:', os.path.basename(out_path))
