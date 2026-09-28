# 2주차 실습 · 단계 6 : 데이터셋 만들기 — 누수(data leakage) 막기
# 이번 주 실습에서 "가장 중요한" 단계라고 슬라이드에서 강조한 부분이다.
#
# 핵심 원칙: 윈도우를 먼저 다 만들어놓고 나중에 train/test로 나누면 안 된다.
#   같은 시행(3초짜리 원본 파일)에서 나온 19개의 윈도우는 서로 150ms만 밀린, 거의 똑같은
#   신호이다. 만약 어떤 시행의 윈도우 일부는 학습 세트에, 나머지는 테스트 세트에
#   들어가면, 모델이 "테스트 세트의 정답"을 사실상 학습 단계에서 이미 봐버린 것과
#   같아진다(=누수). 이러면 테스트 정확도가 비정상적으로(99%+) 높게 나오지만 실제
#   일반화 성능과는 무관한 가짜 점수가 된다.
#   그래서 반드시 "시행(파일) 단위로 먼저" train/test를 나눈 뒤, 그 안에서만 윈도우를
#   만들어야 한다(아래 build() 함수가 그 순서를 지킨다).
#
# 실행 방법: semg-auth\week2 폴더 안에서 step1~5의 함수들을 이 스크립트가 직접 import해서
# 재사용하므로, step1~5를 미리 실행해둘 필요는 없다(파일만 있으면 됨). 그냥:
#   ../.venv/Scripts/python step6_dataset.py
# 주의: 전체 250개 시행 x 19윈도우 = 4750개 조각을 전부 CWT 변환하므로 약 20~30초 걸리고,
#       결과 파일(step6_dataset.npz)이 수백 MB로 크다.

import os
import time
import glob
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# 앞 단계들에서 만든 함수를 그대로 재사용한다. 각 step*.py 파일 맨 아래
# `if __name__ == '__main__':` 블록 덕분에, 이렇게 import만 했을 때는
# 그 안의 실행 코드(파일 읽기/저장/그림 그리기 등)가 실행되지 않고 함수 정의만 가져와진다.
from step1_filter import preprocess
from step3_window import make_windows
from step4_normalize import minmax
from step5_cwt import to_cwt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
SUBJECTS = ['A', 'B', 'C', 'D', 'E']
# 문자열 레이블('A'~'E')을 모델이 다룰 수 있는 정수(0~4)로 바꾸는 매핑.
LABEL_OF_SUBJECT = {s: i for i, s in enumerate(SUBJECTS)}


def load(path):
    """csv 파일 하나를 (3000, 2) 모양의 numpy 배열로 읽는다."""
    return np.loadtxt(path, delimiter=',', skiprows=1)


def subject_of(path):
    """파일 경로에서 피험자 폴더 이름('A'~'E')만 뽑아낸다.
    예: '.../data/data/A/a (3).csv' -> 'A'
    """
    return os.path.basename(os.path.dirname(path))


def build(flist):
    """시행 파일 목록(flist)을 받아서 필터->윈도우->정규화->CWT까지 전부 거친
    (X, y) 데이터셋을 만든다. flist에 들어있는 파일들끼리만 윈도우가 만들어지므로,
    이 함수를 train 파일 목록과 test 파일 목록에 각각 따로 호출하면 두 세트 사이에
    윈도우가 섞일 일이 없다.
    """
    X, y = [], []
    for f in flist:
        # 파일 하나(3초 시행)를 읽어서 필터링한다.
        sig = preprocess(load(f))
        # 필터링된 신호 하나에서 19개의 윈도우를 만들고, 각각 정규화 후 CWT 변환한다.
        for w in minmax(make_windows(sig)):
            X.append(to_cwt(w))
            y.append(LABEL_OF_SUBJECT[subject_of(f)])
    # 리스트를 하나의 큰 numpy 배열로 합친다. X: (N, 3, 32, 300), y: (N,)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


if __name__ == '__main__':
    # 전체 250개 시행 파일 경로와 각 파일의 레이블(피험자)을 먼저 구한다.
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data', '*', '*.csv')))
    labels = [LABEL_OF_SUBJECT[subject_of(f)] for f in files]
    print('전체 시행(파일) 수:', len(files))

    # 1) 시행 단위로 먼저 분할 ← 순서가 핵심 (누수 방지, 위 설명 참고)
    #    stratify=labels : 5명의 피험자가 학습/테스트 양쪽에 같은 비율로 들어가도록 강제한다.
    #                       (안 하면 운 나쁘면 특정 피험자가 테스트에 몰릴 수 있다.)
    #    random_state=42 : 난수 시드를 고정해서, 스크립트를 몇 번을 다시 돌려도
    #                       "항상 같은 분할"이 나오게 한다 (재현성).
    tr_files, te_files = train_test_split(
        files, test_size=0.2, stratify=labels, random_state=42)
    print('학습 시행 수:', len(tr_files), ' 테스트 시행 수:', len(te_files))

    t0 = time.time()
    # 2) 각 집합 안에서만 윈도우 생성 — train 파일들과 test 파일들을 완전히 독립적으로 처리
    Xtr, ytr = build(tr_files)
    Xte, yte = build(te_files)
    print('CWT 변환 소요 시간(초):', round(time.time() - t0, 1))

    print('Xtr:', Xtr.shape, ' Xte:', Xte.shape)
    print('학습 클래스 분포:', Counter(ytr.tolist()))
    print('테스트 클래스 분포:', Counter(yte.tolist()))

    # 시행 단위 분할이 실제로 잘 지켜졌는지 마지막으로 다시 검증한다.
    # tr_files와 te_files는 "파일 경로" 목록이므로, 두 집합(set)의 교집합이 반드시
    # 비어 있어야(길이 0) 같은 시행이 양쪽에 동시에 들어가지 않았다고 확신할 수 있다.
    overlap = set(tr_files) & set(te_files)
    print('학습/테스트 시행 중복 개수(0이어야 정상):', len(overlap))

    # 여러 개의 배열을 한꺼번에 하나의 압축 .npz 파일로 저장한다.
    # savez_compressed는 savez보다 느리지만 파일 크기를 줄여준다(그래도 수백 MB 수준).
    np.savez_compressed(
        os.path.join(BASE_DIR, 'step6_dataset.npz'),
        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
        subjects=np.array(SUBJECTS),
    )
    print('저장 완료: step6_dataset.npz')
