# 2주차 실습 · 단계 10 (추가, 성능 개선 시도 2단계) : 45에폭까지 이어 학습
#
# 왜 이 스크립트가 필요한가:
#   step9_train_more.py(5→20에폭)는 실행하지 않은 채로 45에폭 요청이 들어왔다.
#   20에폭 가중치가 없으므로 20→45로 이어받을 수 없어, 이미 존재하는 5에폭 가중치
#   (step7_model.pt)에서 곧바로 45에폭까지 40에폭을 이어 학습한다 (warm start).
#   원 논문은 45에폭 학습으로 test accuracy 94.00%를 보고했다 — 이 스크립트의 목표는
#   그 설정(에폭 수)을 그대로 맞춰서 재현해보는 것이다.
#
# 실행 방법: semg-auth\week2 폴더 안에서 step7_train.py가 먼저 끝나 있어야 함
#   ../.venv/Scripts/python step10_train_45epoch.py
# 예상 소요 시간: 에폭당 약 361초 x 40에폭 ≈ 4시간
# 결과: step10_model_45epoch.pt (최종 가중치), step10_train_log.txt (1~45에폭 전체 loss 기록)

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
START_EPOCH = 5           # 이미 5에폭까지 학습된 가중치(step7_model.pt)에서 이어서 시작
ADDITIONAL_EPOCHS = 40    # 여기서 40에폭을 더 돌려서 총 45에폭(논문과 동일)을 채운다
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
    # step7에서 이미 5에폭 학습한 가중치를 불러와서 이어서 학습한다 (warm start).
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'step7_model.pt'), map_location=dev))
    model = model.to(dev)

    # 옵티마이저는 새로 만든다 (Adam의 내부 모멘텀 상태는 저장해두지 않았으므로 처음부터
    # 다시 추정하게 되는데, 몇 스텝 안에 다시 안정화되므로 큰 문제는 아니다).
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # 이전 5에폭의 로그를 이어받아서, 최종 로그 파일에 1~45에폭이 전부 보이게 만든다.
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

        # 오래 걸리는 학습이므로 중간에 끊겨도 대비할 수 있게 매 5에폭마다 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(BASE_DIR, f'step10_model_epoch{epoch + 1}.pt'))

    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'step10_model_45epoch.pt'))
    with open(os.path.join(BASE_DIR, 'step10_train_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    print('저장 완료: step10_model_45epoch.pt, step10_train_log.txt')
