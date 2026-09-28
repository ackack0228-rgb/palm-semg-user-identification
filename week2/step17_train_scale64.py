# 2주차 실습 · 단계 17 (추가): CWT 스케일 64 데이터셋으로 학습
#
# step16에서 만든 (3, 64, 300) 텐서로 학습한다. 레시피는 step13(시드 학습)과 동일하게
# weight_decay + CosineAnnealingLR + 약한 증강 + Dropout(0.5)을 그대로 쓴다 — 바뀐 건
# 입력 데이터의 스케일 축 크기(32->64)뿐이므로, DenseNet161은 입력 해상도에 상관없이
# 그대로 쓸 수 있다(global pooling 덕분에 classifier 입력 차원도 안 바뀜).
# 시간축 마스킹 증강은 그대로(최대 30/300), 스케일축 마스킹은 64개 중 최대 8개로
# 비례 조정했다(기존 32개 중 4개와 같은 비율 12.5%).
#
# 사용법: ../.venv/Scripts/python step17_train_scale64.py
# 결과: step17_model_scale64.pt, step17_train_log.txt

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 1
TOTAL_EPOCHS = 45
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
NOISE_STD = 0.02
MASK_TIME_MAX = 30
MASK_SCALE_MAX = 8  # 64개 중 최대 8개 (기존 32개 중 4개와 동일 비율 12.5%)


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

    d = np.load(os.path.join(BASE_DIR, 'step16_dataset_scale64.npz'))
    Xtr = torch.from_numpy(d['Xtr'])
    ytr = torch.from_numpy(d['ytr'])
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, generator=g)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

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
        line = f'epoch {epoch + 1}/{TOTAL_EPOCHS}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s'
        print(line)
        log_lines.append(line)

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'step17_model_scale64.pt'))
    with open(os.path.join(BASE_DIR, 'step17_train_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    print('saved: step17_model_scale64.pt')
