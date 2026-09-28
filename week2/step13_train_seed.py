# 2주차 실습 · 단계 13 (추가, 앙상블용 다중 시드 학습)
#
# step12에서 확인한 사실: step10(88.21%)과 step11(89.05%) 단일 모델을 그냥 평균만
# 내도 91.58%까지 오른다 — 즉 남은 정확도 격차의 상당 부분이 "하이퍼파라미터 문제"가
# 아니라 "작은 테스트셋(피험자당 트라이얼 10개)에서 오는 모델별 무작위 변동성"이다.
# 이를 근거로, 서로 다른 random seed로 모델을 여러 개 더 학습해서 앙상블 후보를
# 늘린다. 각 모델은 step11과 동일한 설정(weight decay + CosineAnnealingLR + 약한
# 증강)을 쓰되 seed만 다르게 한다.
#
# + Dropout 추가 (2026-09-22): 원 논문 Methods에 "정규화: Dropout 적용(과적합 방지)"이
# 명시돼 있는데, step7/step10/step11 전부 분류기가 nn.Linear(2208, 5) 단독이라 이
# 설정이 빠져 있었다. step10/step11 대비 train loss가 0.04~0.07까지 내려가는데
# test 정확도는 88~89%에 머무는 과적합 격차를 줄이기 위해, 이 스크립트부터
# classifier에 Dropout(0.5)을 추가한다(마지막 Linear 직전). 이 변경으로 state_dict
# 키 이름이 'classifier.weight' -> 'classifier.1.weight'로 바뀌므로, 이 체크포인트를
# 평가/앙상블할 때는 build_model(dropout=True)로 만든 모델을 써야 한다
# (step8_evaluate.py / step12_diagnose_ensemble.py 참고).
#
# 사용법: ../.venv/Scripts/python step13_train_seed.py <seed>
# 결과: step13_model_seed<seed>.pt, step13_train_log_seed<seed>.txt

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TOTAL_EPOCHS = 45
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
NOISE_STD = 0.02
MASK_TIME_MAX = 30
MASK_SCALE_MAX = 4


def augment(xb):
    xb = xb + torch.randn_like(xb) * NOISE_STD
    for i in range(xb.shape[0]):
        if random.random() < 0.5:
            t = random.randint(1, MASK_TIME_MAX)
            t0 = random.randint(0, xb.shape[3] - t)
            xb[i, :, :, t0:t0 + t] = 0
        if random.random() < 0.5:
            s = random.randint(1, MASK_SCALE_MAX)
            s0 = random.randint(0, xb.shape[2] - s)
            xb[i, :, s0:s0 + s, :] = 0
    return xb


if __name__ == '__main__':
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    d = np.load(os.path.join(BASE_DIR, 'step6_dataset.npz'))
    Xtr = torch.from_numpy(d['Xtr'])
    ytr = torch.from_numpy(d['ytr'])
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, generator=g)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[seed {SEED}] device:', dev)

    model = densenet161(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
    model = model.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS)
    crit = nn.CrossEntropyLoss()

    log_lines = []
    t_start = time.time()
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        tot = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            xb = augment(xb)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        avg_loss = tot / len(train_loader)
        elapsed = time.time() - t_start
        line = f'[seed {SEED}] epoch {epoch + 1}/{TOTAL_EPOCHS}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s'
        print(line)
        log_lines.append(line)

    torch.save(model.state_dict(), os.path.join(BASE_DIR, f'step13_model_seed{SEED}.pt'))
    with open(os.path.join(BASE_DIR, f'step13_train_log_seed{SEED}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    print(f'[seed {SEED}] 저장 완료: step13_model_seed{SEED}.pt')
