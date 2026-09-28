# 2주차 실습 · 단계 9 (추가, 성능 개선 시도) : 학습을 더 오래 돌려서 정확도 높이기
#
# 왜 이 스크립트가 필요한가:
#   슬라이드 7단계(step7_train.py)는 "학습이 시작되고 loss가 줄어드는 것"만 확인하면
#   되므로 5에폭만 돌렸다. 그 결과로 테스트 정확도를 재보니(step8_evaluate.py) 76.74%로,
#   원 논문의 45에폭 학습 결과(94.00%)에는 한참 못 미쳤다 — 5에폭은 아직 덜 수렴된
#   상태였다는 뜻이다. 딥러닝에서 가장 확실하게 성능을 올리는 방법은 "더 많이 학습시키는
#   것"이므로, 이미 학습된 5에폭 가중치(step7_model.pt)를 이어받아 15에폭을 추가로
#   학습해서 총 20에폭 효과를 낸다. (처음부터 20에폭을 다시 돌리는 것보다, 이미 학습된
#   5에폭 분량을 재사용하면 그만큼 시간을 절약할 수 있다.)
#
# 실행 방법: semg-auth\week2 폴더 안에서 step7_train.py가 먼저 끝나 있어야 함
#   ../.venv/Scripts/python step9_train_more.py
# 예상 소요 시간: 에폭당 약 6분 x 15에폭 ≈ 1.5시간
# 결과: step9_model_20epoch.pt (최종 가중치), step9_train_log.txt (1~20에폭 전체 loss 기록)

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
START_EPOCH = 5           # 이미 5에폭까지 학습된 가중치에서 이어서 시작
ADDITIONAL_EPOCHS = 15    # 여기서 15에폭을 더 돌려서 총 20에폭을 채운다
TOTAL_EPOCHS = START_EPOCH + ADDITIONAL_EPOCHS
BATCH_SIZE = 16

if __name__ == '__main__':
    d = np.load(os.path.join(BASE_DIR, 'step6_dataset.npz'))
    Xtr = torch.from_numpy(d['Xtr'])
    ytr = torch.from_numpy(d['ytr'])
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    model = densenet161(weights=None)
    model.classifier = nn.Linear(2208, 5)
    # ⭐ 처음부터(random init) 시작하지 않고, step7에서 이미 5에폭 학습한 가중치를 불러와서
    # 이어서 학습한다 (warm start). 이렇게 하면 앞서 학습에 쓴 계산을 낭비하지 않는다.
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'step7_model.pt'), map_location=dev))
    model = model.to(dev)

    # 옵티마이저는 새로 만든다 (Adam의 내부 모멘텀 상태는 저장해두지 않았으므로 처음부터
    # 다시 추정하게 되는데, 몇 스텝 안에 다시 안정화되므로 큰 문제는 아니다).
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # 이전 5에폭의 로그를 이어받아서, 최종 로그 파일에 1~20에폭이 전부 보이게 만든다.
    log_lines = []
    prev_log_path = os.path.join(BASE_DIR, 'step7_train_log_5epoch.txt')
    if os.path.exists(prev_log_path):
        with open(prev_log_path, encoding='utf-8') as f:
            log_lines = [line.strip() for line in f if line.strip()]

    t_start = time.time()
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        model.train()
        tot = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        avg_loss = tot / len(train_loader)
        elapsed = time.time() - t_start
        line = f'epoch {epoch + 1}/{TOTAL_EPOCHS}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s (이어학습 구간)'
        print(line)
        log_lines.append(line)

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'step9_model_20epoch.pt'))
    with open(os.path.join(BASE_DIR, 'step9_train_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    print('저장 완료: step9_model_20epoch.pt, step9_train_log.txt')
