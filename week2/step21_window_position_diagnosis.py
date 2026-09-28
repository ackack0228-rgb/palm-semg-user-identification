# 2주차 실습 · 단계 21 (추가, 진단): E 피험자 오분류가 윈도우 위치(시간대)와 관련있는지 확인
#
# 1주차 관찰 노트(week1/observation_notes.md)에 "E는 정지 구간 끝(2.7~2.9s)에 스파이크"
# 라는 메모가 있다. 300ms 윈도우/150ms hop이면 각 시행(3초)에서 19개 윈도우가 나오고,
# i번째(0-index) 윈도우는 [i*150, i*150+300)ms 구간을 담는다 -> i=18(마지막 윈도우)이
# 정확히 [2700, 3000)ms, 즉 스파이크가 보고된 구간과 일치한다. 재학습 없이 기존
# 6-모델 앙상블 예측을 윈도우 위치별로 쪼개서, E의 오분류가 특정 시간대(특히 마지막
# 윈도우 부근)에 쏠리는지 확인한다.

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161
from sklearn.model_selection import train_test_split
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
SUBJECTS = ['A', 'B', 'C', 'D', 'E']
LABEL_OF_SUBJECT = {s: i for i, s in enumerate(SUBJECTS)}
WINDOWS_PER_TRIAL = 19

SCALE32_CHECKPOINTS = [
    'step10_model_45epoch.pt',
    'step11_model_45epoch.pt',
    'step13_model_seed1.pt',
    'step13_model_seed2.pt',
    'step13_model_seed3.pt',
]
SCALE64_CHECKPOINT = 'step17_model_scale64.pt'


def subject_of(path):
    return os.path.basename(os.path.dirname(path))


def build_model(dropout=False):
    model = densenet161(weights=None)
    if dropout:
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
    else:
        model.classifier = nn.Linear(2208, 5)
    return model


if __name__ == '__main__':
    # step6/step16과 완전히 동일한 방식으로 te_files 순서를 재현한다 (seed=42 고정).
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data', '*', '*.csv')))
    labels = [LABEL_OF_SUBJECT[subject_of(f)] for f in files]
    tr_files, te_files = train_test_split(files, test_size=0.2, stratify=labels, random_state=42)

    # 각 테스트 윈도우가 몇 번째 시행의 몇 번째 윈도우인지 미리 계산해둔다.
    window_pos = []
    window_subject = []
    for f in te_files:
        for pos in range(WINDOWS_PER_TRIAL):
            window_pos.append(pos)
            window_subject.append(LABEL_OF_SUBJECT[subject_of(f)])
    window_pos = np.array(window_pos)
    window_subject = np.array(window_subject)

    d32 = np.load(os.path.join(BASE_DIR, 'step6_dataset.npz'))
    d64 = np.load(os.path.join(BASE_DIR, 'step16_dataset_scale64.npz'))
    Xte32 = torch.from_numpy(d32['Xte'])
    Xte64 = torch.from_numpy(d64['Xte'])
    yte = d32['yte']
    assert np.array_equal(yte, window_subject), 'window order mismatch vs npz label order'
    assert np.array_equal(d32['yte'], d64['yte'])

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models32 = []
    for ckpt in SCALE32_CHECKPOINTS:
        m = build_model(dropout='step13' in ckpt).to(dev)
        m.load_state_dict(torch.load(os.path.join(BASE_DIR, ckpt), map_location=dev))
        m.eval()
        models32.append(m)
    model64 = build_model(dropout=True).to(dev)
    model64.load_state_dict(torch.load(os.path.join(BASE_DIR, SCALE64_CHECKPOINT), map_location=dev))
    model64.eval()

    loader32 = DataLoader(TensorDataset(Xte32, torch.from_numpy(yte)), batch_size=32, shuffle=False)
    loader64 = DataLoader(TensorDataset(Xte64, torch.from_numpy(yte)), batch_size=32, shuffle=False)

    all_pred = []
    with torch.no_grad():
        for (xb32, yb), (xb64, _) in zip(loader32, loader64):
            xb32, xb64 = xb32.to(dev), xb64.to(dev)
            probs_sum = None
            for m in models32:
                probs = torch.softmax(m(xb32), dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs_sum = probs_sum + torch.softmax(model64(xb64), dim=1)
            pred = probs_sum.argmax(dim=1)
            all_pred.append(pred.cpu().numpy())
    all_pred = np.concatenate(all_pred)

    is_e = window_subject == LABEL_OF_SUBJECT['E']
    correct = (all_pred == yte)

    lines = ['E subject accuracy by window position (0=0.0-0.3s ... 18=2.7-3.0s):']
    for pos in range(WINDOWS_PER_TRIAL):
        mask = is_e & (window_pos == pos)
        n = mask.sum()
        c = correct[mask].sum()
        t_start = pos * 0.15
        lines.append(f'  pos {pos:2d} [{t_start:.2f}-{t_start+0.3:.2f}s]: {c}/{n} ({c/n*100:.1f}%)')

    print('\n'.join(lines))
    with open(os.path.join(BASE_DIR, 'step21_window_position_diagnosis.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved: step21_window_position_diagnosis.txt')
