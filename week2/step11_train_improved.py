# 2주차 실습 · 단계 11 (추가, 성능 개선 시도 3단계) : 목표 96%
#
# 왜 이 스크립트가 필요한가:
#   step10(5->45 웜스타트)로 88.21%까지 올렸지만 논문(94.00%)에는 못 미쳤다. 사용자가 준
#   가설(과적합 + lr 고정 + 증강 없음)을 바탕으로 아래 3가지를 추가해서 다시 시도한다.
#   1) weight decay(1e-4): train loss가 0.04~0.07까지 내려갔는데 test는 88%대였던
#      것을 보면 과적합 신호로 판단, Adam에 weight decay를 추가해 억제한다.
#   2) CosineAnnealingLR: 45에폭 내내 lr=1e-3 고정 대신, 후반으로 갈수록 lr을 서서히
#      줄여 더 정교하게 수렴시킨다.
#   3) CWT 텐서에 SpecAugment 스타일 증강(약한 가우시안 노이즈 + 시간축/스케일축 마스킹):
#      학습 데이터(3800개)가 많지 않아 특정 피험자(B, E)에 덜 강건했던 것으로 보여,
#      매 배치 학습 데이터를 살짝 다르게 만들어 일반화 성능을 높인다.
#   ※ 이 3가지는 원 논문의 정확한 설정에는 없는 추가 시도이다 — "논문 재현"이 아니라
#     "논문보다 더 높은 정확도(목표 96%)"를 사용자가 명시적으로 요청해서 적용함.
#
#   GPU(RTX 3060)가 에폭당 ~34초로 매우 빨라졌기 때문에, 5에폭 가중치를 이어받는 대신
#   처음부터(random init) 45에폭을 전부 새 설정으로 다시 돈다 (~25분 소요, 옵티마이저
#   상태 불일치 문제도 없어져서 더 깔끔함).
#
# 실행 방법: semg-auth\week2 폴더 안에서
#   ../.venv/Scripts/python step11_train_improved.py
# 예상 소요 시간: 에폭당 약 34초 x 45에폭 ≈ 25분
# 결과: step11_model_45epoch.pt (최종 가중치), step11_train_log.txt (1~45에폭 전체 loss/lr 기록)

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOTAL_EPOCHS = 45
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
NOISE_STD = 0.02      # CWT 텐서 값 스케일 대비 약한 노이즈
MASK_TIME_MAX = 30    # 시간축 300 중 최대 10%까지 마스킹
MASK_SCALE_MAX = 4    # 스케일축 32 중 최대 12%까지 마스킹


def augment(xb):
    """CWT 텐서(B, 3, 32, 300)에 SpecAugment 스타일 증강을 적용한다.
    시간 도메인 원신호를 다시 CWT로 변환하는 대신, 이미 계산해둔 CWT 결과에
    직접 노이즈/마스킹을 가해 매 배치마다 조금씩 다른 입력을 만든다."""
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
    d = np.load(os.path.join(BASE_DIR, 'step6_dataset.npz'))
    Xtr = torch.from_numpy(d['Xtr'])
    ytr = torch.from_numpy(d['ytr'])
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    model = densenet161(weights=None)
    model.classifier = nn.Linear(2208, 5)
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
        lr_now = sched.get_last_lr()[0]
        line = f'epoch {epoch + 1}/{TOTAL_EPOCHS}  loss={avg_loss:.4f}  lr={lr_now:.6f}  elapsed={elapsed:.1f}s'
        print(line)
        log_lines.append(line)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(BASE_DIR, f'step11_model_epoch{epoch + 1}.pt'))

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'step11_model_45epoch.pt'))
    with open(os.path.join(BASE_DIR, 'step11_train_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    print('저장 완료: step11_model_45epoch.pt, step11_train_log.txt')
