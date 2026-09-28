# 2주차 실습 · 단계 5 : CWT 변환 (1차원 신호 -> 2차원 시간-주파수 그림)
# 목적: 정규화까지 끝낸 300ms짜리 1차원 신호 조각을, CNN이 이미지처럼 받아들일 수 있는
#       (3, 32, 300) 모양의 3차원 텐서로 바꾼다.
# 실행 방법: semg-auth\week2 폴더 안에서 step1~4를 먼저 실행한 뒤
#   ../.venv/Scripts/python step5_cwt.py
# 결과: step5_t.npy (텐서 1개 예시), cwt_example.png (눈으로 확인할 그림, 제출물 ②)

import os
import numpy as np
import pywt  # PyWavelets: 웨이블릿 변환 라이브러리
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 스케일(scale)은 웨이블릿을 얼마나 늘리거나 줄이는지를 나타내는 값이다.
# 스케일이 작으면 촘촘한(고주파) 성분을, 크면 넓은(저주파) 성분을 잡아낸다.
# 1~32까지 32개의 스케일을 쓰기로 했으므로(논문/슬라이드 설정과 동일), 결과 이미지의
# 세로축(주파수 방향) 픽셀 수가 32가 된다.
SCALES = np.arange(1, 33)  # 스케일 32개 (1, 2, ..., 32)


def to_cwt(one_window, wavelet='morl'):
    """(300, 2) 모양의 윈도우 1개 -> (3, 32, 300) 모양의 CNN 입력 텐서로 변환.

    wavelet='morl' (Morlet 웨이블릿)을 쓰는 이유: 근전도(sEMG)처럼 진동하며 감쇠하는
    신호의 시간-주파수 특성을 볼 때 흔히 쓰이는 마더 웨이블릿이기 때문.
    """
    maps = []
    # 채널 수(2개: APB, ADM)만큼 반복하면서 각 채널을 따로 CWT 변환한다.
    for ch in range(one_window.shape[1]):
        # pywt.cwt(1차원 신호, 스케일 배열, 웨이블릿 이름)
        # -> coef: (스케일 수, 시간 길이) = (32, 300) 모양의 복소수 계수 배열
        coef, _ = pywt.cwt(one_window[:, ch], SCALES, wavelet)
        # 복소수 계수의 절댓값(진폭)만 취해서 실수 이미지로 만든다.
        maps.append(np.abs(coef))  # (32, 300)
    # CNN(DenseNet161 등)은 보통 3채널(RGB) 입력을 기대하는데 우리는 2채널(APB, ADM)뿐이라,
    # 3번째 채널을 앞의 두 채널 CWT 결과의 평균으로 채워서 3채널 입력 형식을 맞춘다.
    maps.append((maps[0] + maps[1]) / 2)  # 3번째 채널
    # (3, 32, 300) 모양으로 쌓고, float32로 변환해서 파이토치가 바로 쓸 수 있게 한다.
    return np.stack(maps).astype(np.float32)


if __name__ == '__main__':
    # 정규화까지 끝낸 윈도우들 중 첫 번째(wn[0]) 하나만 예시로 변환해본다.
    # (250개 시행 전체를 CWT 변환하는 건 6단계 데이터셋 생성에서 한다.)
    wn = np.load(os.path.join(BASE_DIR, 'step4_wn.npy'))
    t = to_cwt(wn[0])
    print('입력 텐서 모양:', t.shape)  # (3, 32, 300) 이 나와야 정상
    np.save(os.path.join(BASE_DIR, 'step5_t.npy'), t)

    # imshow로 3채널 각각을 그림으로 그려서 실제로 시간-주파수 패턴이 보이는지 확인한다.
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    titles = ['ch1 (APB)', 'ch2 (ADM)', 'ch1+ch2 avg']
    for i in range(3):
        # origin='lower' : 배열의 0번째 행(작은 스케일=고주파)이 그림의 아래쪽에 오게 한다.
        # aspect='auto'  : 가로세로 비율을 억지로 1:1로 맞추지 않고 칸에 맞게 늘린다.
        im = ax[i].imshow(t[i], aspect='auto', origin='lower', cmap='jet')
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('time (samples)')
        ax[i].set_ylabel('scale')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'cwt_example.png'), dpi=120)
    print('저장 완료: step5_t.npy, cwt_example.png')
