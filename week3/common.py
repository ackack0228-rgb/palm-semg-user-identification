# 3주차 공통 모듈 — 모든 모델이 "같은 조건"으로 학습되도록 레시피를 한 곳에 모았다.
#
# 3주차 슬라이드 단계 3: "반드시 같은 분할·같은 에폭·같은 학습률로. 하나라도 조건이
# 다르면 비교가 성립하지 않는다." → 모델 구조만 바꾸고 나머지(데이터 분할, 45에폭,
# Adam lr=1e-3, weight decay 1e-4, CosineAnnealingLR, 증강, Dropout 0.5, seed)는 전부
# 이 파일의 train_model() 하나로 통일한다. 이 레시피는 2주차 step13_train_seed.py
# (DenseNet161, seed=1 → 테스트 91.37%)와 완전히 동일하다.

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161, resnet18

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEEK2_DIR = os.path.join(BASE_DIR, '..', 'week2')
sys.path.insert(0, WEEK2_DIR)  # 2주차 전처리 함수(step1~6) 재사용

SUBJECTS = ['A', 'B', 'C', 'D', 'E']
TOTAL_EPOCHS = 45
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
NOISE_STD = 0.02
MASK_TIME_MAX = 30
MASK_SCALE_MAX = 4
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_model(name):
    """name: 'cnn2d' | 'resnet18' | 'densenet161'. 세 모델 모두 사전학습 없이(weights=None)
    처음부터 학습하고, 마지막 분류기는 Dropout(0.5) + Linear(→5)로 통일한다."""
    if name == 'cnn2d':
        # 3주차 슬라이드 단계 3의 "베이스라인 A: 간단한 2D CNN" 구조 그대로
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, 5))
    if name == 'resnet18':
        m = resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 5))
        return m
    if name == 'densenet161':
        m = densenet161(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
        return m
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def augment(xb):
    """가우시안 노이즈 + 시간축/스케일축 랜덤 마스킹 (2주차 step11/13과 동일)."""
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


def train_model(model, Xtr, ytr, seed=1, tag=''):
    """model을 (Xtr, ytr)로 45에폭 학습하고 (model, 학습시간(초), loss 로그)를 반환."""
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.as_tensor(Xtr), torch.as_tensor(ytr)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=g)
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_EPOCHS)
    crit = nn.CrossEntropyLoss()
    log = []
    t0 = time.time()
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        tot = 0
        for xb, yb in loader:
            xb, yb = augment(xb.to(DEV)), yb.to(DEV).long()
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        line = f'{tag} epoch {epoch + 1}/{TOTAL_EPOCHS}  loss={tot / len(loader):.4f}  elapsed={time.time() - t0:.1f}s'
        print(line, flush=True)
        log.append(line)
    return model, time.time() - t0, log


@torch.no_grad()
def predict_proba(model, X, batch=64):
    """softmax 확률 (N, 5) 반환. model.eval()을 반드시 호출한다(Dropout/BN 고정)."""
    model.eval().to(DEV)
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch]).to(DEV)
        out.append(torch.softmax(model(xb), 1).cpu())
    return torch.cat(out).numpy()
