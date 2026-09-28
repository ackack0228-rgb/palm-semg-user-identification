# 2주차 실습 · 단계 16 (추가): CWT 스케일 32 -> 64로 늘린 데이터셋 재생성
#
# step15 혼동행렬에서 E 피험자 오분류의 71%(27/38)가 B로 쏠리는 걸 확인했다. 5개
# 모델(다른 초기화/증강)을 앙상블해도 이 편향이 거의 그대로인 걸 보면, 무작위 변동성이
# 아니라 "CWT 스케일 32개짜리 시간-주파수 표현 자체가 E와 B의 근전도 패턴을 잘 구분하지
# 못한다"는 특징 표현(feature representation)의 한계일 가능성이 크다. 3주차 PDF
# "심화 도전 과제"에 공식적으로 나열된 방향 중 하나인 CWT 스케일 수를 64개로 늘려서
# (더 촘촘한 주파수 해상도) 다시 데이터셋을 만들어 이 가설을 검증한다.
#
# step5_cwt.py/step6_dataset.py는 기존 32-스케일 파이프라인이 계속 재현 가능하도록
# 건드리지 않고, 이 스크립트에서 SCALES만 바꾼 버전을 독립적으로 둔다.
#
# 실행 방법: semg-auth\week2 폴더 안에서
#   ../.venv/Scripts/python step16_dataset_scale64.py
# 결과: step16_dataset_scale64.npz (Xtr/ytr/Xte/yte, 모양 (N, 3, 64, 300))

import os
import time
import glob
import numpy as np
import pywt
from collections import Counter
from sklearn.model_selection import train_test_split

from step1_filter import preprocess
from step3_window import make_windows
from step4_normalize import minmax

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
SUBJECTS = ['A', 'B', 'C', 'D', 'E']
LABEL_OF_SUBJECT = {s: i for i, s in enumerate(SUBJECTS)}

SCALES = np.arange(1, 65)  # 스케일 64개 (기존 32개의 2배)


def to_cwt64(one_window, wavelet='morl'):
    """(300, 2) -> (3, 64, 300). step5_cwt.to_cwt와 동일한 구조, 스케일 수만 64."""
    maps = []
    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(one_window[:, ch], SCALES, wavelet)
        maps.append(np.abs(coef))
    maps.append((maps[0] + maps[1]) / 2)
    return np.stack(maps).astype(np.float32)


def load(path):
    return np.loadtxt(path, delimiter=',', skiprows=1)


def subject_of(path):
    return os.path.basename(os.path.dirname(path))


def build(flist):
    X, y = [], []
    for f in flist:
        sig = preprocess(load(f))
        for w in minmax(make_windows(sig)):
            X.append(to_cwt64(w))
            y.append(LABEL_OF_SUBJECT[subject_of(f)])
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data', '*', '*.csv')))
    labels = [LABEL_OF_SUBJECT[subject_of(f)] for f in files]
    print('total trial files:', len(files))

    # step6과 동일한 split(seed=42)을 그대로 써서 기존 결과와 비교 가능하게 한다.
    tr_files, te_files = train_test_split(
        files, test_size=0.2, stratify=labels, random_state=42)
    print('train trials:', len(tr_files), ' test trials:', len(te_files))

    t0 = time.time()
    Xtr, ytr = build(tr_files)
    Xte, yte = build(te_files)
    print('CWT elapsed(s):', round(time.time() - t0, 1))

    print('Xtr:', Xtr.shape, ' Xte:', Xte.shape)
    print('train class dist:', Counter(ytr.tolist()))
    print('test class dist:', Counter(yte.tolist()))

    overlap = set(tr_files) & set(te_files)
    print('train/test overlap (should be 0):', len(overlap))

    np.savez_compressed(
        os.path.join(BASE_DIR, 'step16_dataset_scale64.npz'),
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
        subjects=np.array(SUBJECTS),
    )
    print('saved: step16_dataset_scale64.npz')
