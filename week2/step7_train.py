# 2주차 실습 · 단계 7 : 학습 실행 (오늘의 마지막 단계)
# 목적: DenseNet161을 불러와서, 6단계에서 만든 데이터셋으로 실제로 학습이 시작되고
#       loss(손실)가 출력되며 줄어드는지 확인한다. (45에폭 전체 학습은 별도로 시간 날 때
#       돌리면 되고, 오늘 제출용으로는 최소 5에폭의 loss 기록만 있으면 된다.)
#
# 실행 방법: semg-auth\week2 폴더 안에서 step6_dataset.py를 먼저 실행해 둔 뒤
#   ../.venv/Scripts/python step7_train.py
# 주의: CPU로 돌리면 DenseNet161 특성상 에폭당 5분 안팎이 걸릴 수 있어, 5에폭이면
#       총 25~30분 정도 예상된다. 오래 걸리므로 백그라운드로 실행하는 것을 권장.
# 결과: step7_model.pt (학습된 가중치), step7_train_log.txt (에폭별 loss 기록, 제출물 ⑤)

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 5  # 오늘은 5에폭만 (과제 요구: 최소 5에폭 loss 기록). 45에폭 전체 학습은 별도로 진행.
BATCH_SIZE = 16  # 한 번에 16개 샘플씩 묶어서 학습. CUDA 메모리 부족하면 8이나 4로 줄이면 됨.

if __name__ == '__main__':
    # 6단계에서 저장해둔 압축 데이터셋을 불러온다. (Xtr: 학습 입력, ytr: 학습 정답)
    d = np.load(os.path.join(BASE_DIR, 'step6_dataset.npz'))
    # numpy 배열을 파이토치 텐서로 변환한다. (아직 CPU 텐서 상태, GPU로는 배치마다 옮김)
    Xtr = torch.from_numpy(d['Xtr'])
    ytr = torch.from_numpy(d['ytr'])

    # TensorDataset : (입력, 정답) 쌍을 인덱스로 꺼낼 수 있게 묶어주는 래퍼.
    # DataLoader    : 그 데이터셋을 배치 단위로 잘라주고, shuffle=True로 매 에폭마다
    #                 순서를 섞어서 모델이 데이터 순서를 외우지 못하게 한다.
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)

    # GPU(CUDA)가 있으면 GPU를, 없으면 CPU를 자동으로 선택한다.
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev)

    # weights=None : ImageNet 등으로 사전학습된 가중치를 쓰지 않고 처음부터(random init) 학습.
    # 우리 입력(3x32x300, sEMG의 CWT 스펙트로그램)이 사전학습에 쓰인 자연 이미지와는
    # 성질이 많이 달라서, 사전학습 가중치를 그대로 재사용하는 이점이 크지 않기 때문.
    model = densenet161(weights=None)  # 사전학습 미사용
    # DenseNet161의 원래 마지막 층은 ImageNet의 1000개 클래스를 분류하도록 되어 있는데,
    # 우리는 "5명 중 누구인지"만 맞히면 되므로 출력 차원을 5로 바꾼 새 Linear 층으로 교체한다.
    # 2208은 DenseNet161의 마지막 특징맵 채널 수(고정값, 모델 구조에서 정해짐).
    model.classifier = nn.Linear(2208, 5)  # 5명 분류
    model = model.to(dev)  # 모델 파라미터를 선택한 장치(CPU/GPU)로 옮긴다.

    # Adam : 손실을 줄이는 방향으로 가중치를 갱신하는 옵티마이저. 가장 무난한 기본 선택.
    # lr=1e-3(0.001) : 한 번에 얼마나 크게 가중치를 고칠지. 너무 크면 발산, 작으면 너무 느리다.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # CrossEntropyLoss : 다중 클래스 분류(5명 중 하나 맞히기)에서 표준적으로 쓰는 손실 함수.
    crit = nn.CrossEntropyLoss()

    log_lines = []  # 에폭마다 한 줄씩 기록을 모아뒀다가 마지막에 파일로 저장한다.
    t_start = time.time()
    for epoch in range(EPOCHS):
        model.train()  # 학습 모드로 전환 (Dropout/BatchNorm이 학습용으로 동작)
        tot = 0
        for xb, yb in train_loader:  # 배치 하나(16개 샘플)씩 꺼낸다
            xb, yb = xb.to(dev), yb.to(dev)  # 이 배치만 선택한 장치로 옮긴다 (메모리 절약)
            loss = crit(model(xb), yb)       # 순전파: 예측하고 정답과 비교해 손실을 계산
            # 역전파 3단계: 이전 gradient 초기화 -> gradient 계산 -> 가중치 업데이트
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()  # .item()으로 텐서를 순수 파이썬 숫자로 꺼내서 누적
        avg_loss = tot / len(train_loader)  # 이번 에폭의 배치별 loss 평균
        elapsed = time.time() - t_start
        line = f'epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s'
        print(line)  # loss가 에폭이 지날수록 줄어드는지 눈으로 확인
        log_lines.append(line)

    # 학습된 가중치를 저장해둔다 — 다음 주(3주차) 실습에서 이 가중치로 테스트 정확도,
    # 혼동행렬 등을 계산할 때 다시 불러와 쓸 수 있다.
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'step7_model.pt'))
    # 제출물 ⑤(학습 로그)로 쓸 수 있도록 에폭별 loss를 텍스트 파일로도 남긴다.
    with open(os.path.join(BASE_DIR, 'step7_train_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    print('저장 완료: step7_model.pt, step7_train_log.txt')
