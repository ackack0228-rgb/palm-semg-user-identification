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
from torchvision.models import densenet161, resnet18, convnext_tiny

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


RAW_INPUT_MODELS = {'cnn1d'}  # CWT가 아니라 원 신호 윈도우 (2, 300)를 입력으로 받는 모델


class _ConvFront(nn.Module):
    """CWT (3, 32, 300) → 시간축 시퀀스 (B, 75, 128). 순환 모델(CNN-BiLSTM, CNN-LSTM-Attention)의
    CNN 부분: 합성곱으로 지역 특징을 뽑은 뒤 스케일(주파수)축만 평균내고 시간축은 시퀀스로 남긴다."""
    def __init__(self):
        super().__init__()
        def blk(i, o, pool):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
            return layers + ([nn.MaxPool2d(2)] if pool else [])
        self.net = nn.Sequential(*blk(3, 32, True), *blk(32, 64, True), *blk(64, 128, False),
                                 nn.AdaptiveAvgPool2d((1, None)))   # (B, 128, 1, 75)

    def forward(self, x):
        return self.net(x).squeeze(2).transpose(1, 2)               # (B, 75, 128)


class CNNBiLSTM(nn.Module):
    """논문 참고문헌 22(Kishore et al.): CNN으로 지역 특징 → BiLSTM으로 양방향 시간 의존성."""
    def __init__(self, hidden=192):
        super().__init__()
        self.front = _ConvFront()
        self.lstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(2 * hidden, 5))

    def forward(self, x):
        out, _ = self.lstm(self.front(x))
        return self.head(out.mean(1))


class CNNLSTMAttention(nn.Module):
    """논문 참고문헌 23(Hwang et al.): LSTM 출력의 시간 구간별 중요도(attention 가중치)를 학습해
    가중합한다 — "모든 시간 구간이 같은 식별 정보를 담고 있지 않다"는 가정."""
    def __init__(self, hidden=128):
        super().__init__()
        self.front = _ConvFront()
        self.lstm = nn.LSTM(128, hidden, batch_first=True)
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden, 5))

    def forward(self, x):
        out, _ = self.lstm(self.front(x))               # (B, T, H)
        w = torch.softmax(self.score(out), dim=1)       # (B, T, 1) 시간 구간별 중요도
        return self.head((w * out).sum(1))


class TriCCNN(nn.Module):
    """논문 참고문헌 25(Tri-CCNN)의 근사 재구현: 세 개의 병렬 합성곱 스트림(multi-stream).
    원 논문은 서로 다른 시간-주파수 표현 3개를 각 스트림에 넣으므로, 여기서는 입력의 세 채널
    (APB CWT, ADM CWT, 두 채널 평균)을 스트림 하나씩에 넣고 마지막에 이어붙인다. 파라미터 수가
    매우 작은 경량 구조(논문 1,357개)라는 특징을 따랐다."""
    def __init__(self, width=8):
        super().__init__()
        def stream():
            return nn.Sequential(
                nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.streams = nn.ModuleList([stream() for _ in range(3)])
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(3 * width, 5))

    def forward(self, x):
        return self.head(torch.cat([s(x[:, i:i + 1]) for i, s in enumerate(self.streams)], 1))


class E2CNN(nn.Module):
    """논문 참고문헌 24(E2CNN)의 근사 재구현: 얕은 층부터 깊은 층까지 여러 단계의 특징을 각각
    GAP해서 이어붙이는(concatenation) 경량 CNN. 원 논문 입력은 Log-Mel 스펙트로그램이지만
    "같은 전처리 조건" 비교를 위해 여기서는 다른 모델과 같은 CWT 입력을 쓴다."""
    def __init__(self):
        super().__init__()
        chans = [3, 16, 32, 64, 96]
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(chans[k], chans[k + 1], 3, padding=1), nn.BatchNorm2d(chans[k + 1]),
                          nn.ReLU(), nn.MaxPool2d(2) if k < 3 else nn.Identity())
            for k in range(4)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(sum(chans[1:]), 5))

    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = b(x)
            feats.append(self.gap(x).flatten(1))
        return self.head(torch.cat(feats, 1))


def build_model(name):
    """name: 'cnn2d' | 'resnet18' | 'densenet161' | 'cnn1d' | 'cnn_bilstm' | 'cnn_lstm_attn' |
    'convnext' | 'tri_ccnn' | 'e2cnn'. 전부 사전학습 없이(weights=None) 처음부터 학습하고,
    마지막 분류기는 Dropout(0.5) + Linear(→5)로 통일한다.
    cnn1d만 원 신호 윈도우 (2, 300)를 받고, 나머지는 CWT (3, 32, 300)를 받는다."""
    if name == 'cnn1d':
        # 3주차 강의 37쪽 "1D CNN — 시계열 직접 입력, 기존 sEMG 식별 연구의 기본형".
        # 논문 Table 4의 파라미터 규모(9,589개)에 맞춘 3층 경량 1D CNN.
        def blk(i, o, k, pool):
            return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()] + \
                   ([nn.MaxPool1d(2)] if pool else [])
        return nn.Sequential(*blk(2, 16, 7, True), *blk(16, 32, 5, True), *blk(32, 64, 3, False),
                             nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, 5))
    if name == 'cnn_bilstm':
        return CNNBiLSTM()
    if name == 'cnn_lstm_attn':
        return CNNLSTMAttention()
    if name == 'tri_ccnn':
        return TriCCNN()
    if name == 'e2cnn':
        return E2CNN()
    if name == 'convnext':
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, 5))
        return m
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


def build_raw_dataset():
    """1D CNN용 원 신호 데이터셋: 2주차 step6와 똑같은 시행 분할(8:2, stratify, seed=42)·
    필터·윈도우·min-max 정규화를 거치고 CWT만 하지 않은 (N, 2, 300) 텐서.
    윈도우 순서가 step6_dataset.npz와 같으므로 레이블이 완전히 일치해야 한다(assert로 확인)."""
    import glob
    from sklearn.model_selection import train_test_split
    from step1_filter import preprocess
    from step3_window import make_windows
    from step4_normalize import minmax
    from step6_dataset import load, subject_of, LABEL_OF_SUBJECT, DATA_DIR
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data', '*', '*.csv')))
    labels = [LABEL_OF_SUBJECT[subject_of(f)] for f in files]
    tr_files, te_files = train_test_split(files, test_size=0.2, stratify=labels, random_state=42)

    def build(flist):
        X, y = [], []
        for f in flist:
            for w in minmax(make_windows(preprocess(load(f)))):
                X.append(w.T)                                   # (300, 2) → (2, 300)
                y.append(LABEL_OF_SUBJECT[subject_of(f)])
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
    return build(tr_files) + build(te_files)


def load_split(name):
    """모델 이름에 맞는 입력으로 (Xtr, ytr, Xte, yte)를 돌려준다. 분할은 모두 step6와 동일."""
    d = np.load(os.path.join(WEEK2_DIR, 'step6_dataset.npz'))
    if name not in RAW_INPUT_MODELS:
        return d['Xtr'], d['ytr'], d['Xte'], d['yte']
    path = os.path.join(BASE_DIR, 'raw_dataset.npz')
    if not os.path.exists(path):
        Xtr, ytr, Xte, yte = build_raw_dataset()
        np.savez_compressed(path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    r = np.load(path)
    assert np.array_equal(r['ytr'], d['ytr']) and np.array_equal(r['yte'], d['yte'])
    return r['Xtr'], r['ytr'], r['Xte'], r['yte']


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def augment(xb):
    """가우시안 노이즈 + 시간축/스케일축 랜덤 마스킹 (2주차 step11/13과 동일).
    원 신호 입력 (B, 2, 300)이면 스케일축이 없으므로 노이즈 + 시간축 마스킹만 한다."""
    xb = xb + torch.randn_like(xb) * NOISE_STD
    if xb.dim() == 3:
        for i in range(xb.shape[0]):
            if random.random() < 0.5:
                t = random.randint(1, MASK_TIME_MAX)
                t0 = random.randint(0, xb.shape[2] - t)
                xb[i, :, t0:t0 + t] = 0
        return xb
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
