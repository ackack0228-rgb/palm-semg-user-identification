# 2주차 실습 · 단계 8 (추가) : 테스트 정확도 평가
# 원래 3주차 실습에서 다룰 내용이지만, "성능을 더 높일 수 있으면 적용해달라"는 요청에
# 따라 에폭 수를 늘리는 게 실제로 효과가 있는지 "숫자로" 확인하기 위해 미리 만들었다.
#
# 사용법 (semg-auth\week2 폴더 안에서):
#   ../.venv/Scripts/python step8_evaluate.py step7_model_5epoch.pt
#   ../.venv/Scripts/python step8_evaluate.py step9_model_20epoch.pt
# 인자를 안 주면 기본값으로 step7_model.pt를 평가한다.

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


def build_model(dropout=False):
    """학습 스크립트와 완전히 동일한 구조로 모델을 만든다.
    (구조가 조금이라도 다르면 저장된 가중치를 불러올 때 오류가 난다.)
    step13(2026-09-22 Dropout 추가) 이후 체크포인트는 dropout=True로 불러와야 한다."""
    model = densenet161(weights=None)
    if dropout:
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
    else:
        model.classifier = nn.Linear(2208, 5)
    return model


@torch.no_grad()  # 평가는 역전파가 필요 없으니 gradient 계산을 꺼서 속도를 높이고 메모리를 아낀다.
def evaluate(model, X, y, dev, batch_size=32):
    model.eval()  # 평가 모드 (Dropout 등을 비활성화)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    correct_total = 0
    n_total = 0
    # 피험자(클래스)별로 맞은 개수/전체 개수를 따로 세서 특정 피험자만 유독 못 맞히는지도 본다.
    correct_per_class = Counter()
    total_per_class = Counter()
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        pred = model(xb).argmax(dim=1)  # 5개 클래스 점수 중 가장 높은 것을 예측으로 선택
        correct = (pred == yb)
        correct_total += correct.sum().item()
        n_total += len(yb)
        for label, is_correct in zip(yb.tolist(), correct.tolist()):
            total_per_class[label] += 1
            correct_per_class[label] += int(is_correct)
    return correct_total / n_total, correct_per_class, total_per_class


if __name__ == '__main__':
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else 'step7_model.pt'
    ckpt_path = os.path.join(BASE_DIR, ckpt_name)

    d = np.load(os.path.join(BASE_DIR, 'step6_dataset.npz'))
    Xte = torch.from_numpy(d['Xte'])
    yte = torch.from_numpy(d['yte'])

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # step13 계열 체크포인트(seed 모델)는 classifier에 Dropout이 들어가 있으므로 구분해서 불러온다.
    model = build_model(dropout='step13' in ckpt_name).to(dev)
    # map_location=dev : GPU에서 학습한 가중치를 CPU에서 불러올 때(또는 그 반대) 오류가
    # 나지 않도록, 저장된 텐서들을 현재 사용 중인 장치로 자동 변환해서 불러온다.
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))

    acc, correct_per_class, total_per_class = evaluate(model, Xte, yte, dev)
    print(f'체크포인트: {ckpt_name}')
    print(f'전체 테스트 정확도: {acc*100:.2f}%  ({sum(correct_per_class.values())}/{sum(total_per_class.values())})')
    print('피험자별 정확도:')
    for i, s in enumerate(SUBJECTS):
        c, t = correct_per_class[i], total_per_class[i]
        print(f'  {s}: {c}/{t} ({c/t*100:.1f}%)')

    out_path = os.path.join(BASE_DIR, f'step8_eval_{os.path.splitext(ckpt_name)[0]}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'checkpoint: {ckpt_name}\n')
        f.write(f'test_accuracy: {acc*100:.2f}%\n')
        for i, s in enumerate(SUBJECTS):
            c, t = correct_per_class[i], total_per_class[i]
            f.write(f'{s}: {c}/{t} ({c/t*100:.1f}%)\n')
    print('저장 완료:', os.path.basename(out_path))
