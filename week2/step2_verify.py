# 2주차 실습 · 단계 2 : 필터가 제대로 걸렸는지 확인
# 목적: step1에서 저장해둔 필터 전/후 신호를 불러와, 주파수 스펙트럼을 그려서
#       60Hz 자리의 봉우리가 실제로 줄었는지 "눈으로" 그리고 "숫자로" 확인한다.
# 실행 방법: semg-auth\week2 폴더 안에서 step1_filter.py를 먼저 실행한 뒤
#   ../.venv/Scripts/python step2_verify.py
# 결과: filter_check.png (리포트에 반드시 넣어야 하는 필수 제출 그림)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 화면 창 없이 파일로만 저장 (서버/터미널 환경 대응)
import matplotlib.pyplot as plt
from scipy.signal import welch  # Welch 방법: 신호를 여러 구간으로 나눠 평균낸 안정적인 파워 스펙트럼 추정

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 1000

# step1_filter.py가 저장해둔 필터 전(x)/후(xf) 신호를 불러온다.
# (단계별로 파일에 저장 -> 다음 스크립트가 그 파일을 불러오는 방식이라, 중간에 막혀도
#  1단계부터 다시 돌릴 필요 없이 여기서부터 이어갈 수 있다.)
x = np.load(os.path.join(BASE_DIR, 'step1_x.npy'))
xf = np.load(os.path.join(BASE_DIR, 'step1_xf.npy'))

# welch(신호, 샘플링주파수, nperseg=한 구간의 길이)
# -> (주파수 배열 f, 그 주파수에서의 파워 배열 P) 를 반환한다.
# ch1(인덱스 0)만 대표로 확인한다. (ch2도 똑같이 필터가 걸리므로 하나만 봐도 충분)
f0, P0 = welch(x[:, 0], fs=FS, nperseg=512)
f1, P1 = welch(xf[:, 0], fs=FS, nperseg=512)

plt.figure(figsize=(9, 3))
# semilogy: y축(파워)만 로그 스케일로 그려서, 작은 값과 큰 값 차이를 한 그래프에서 잘 보이게 한다.
plt.semilogy(f0, P0, label='before', lw=0.8)
plt.semilogy(f1, P1, label='after', lw=0.8)
plt.axvline(60, color='r', ls='--')  # 60Hz 위치를 빨간 점선으로 표시 (여기 봉우리가 사라져야 정상)
plt.xlim(0, 300)
plt.legend()
plt.xlabel('Hz')
plt.ylabel('power')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'filter_check.png'), dpi=120)

# 그림만으로는 "얼마나" 줄었는지 애매할 수 있어서, 58~62Hz 구간의 최대 파워값을
# 필터 전/후로 직접 비교하는 수치 검증도 함께 한다.
band = (f0 >= 58) & (f0 <= 62)
p60_before = P0[band].max()
p60_after = P1[band].max()
print('60Hz 근처 최대 파워 - 전:', p60_before, ' 후:', p60_after)
print('감소 여부:', p60_after < p60_before)  # True가 나와야 필터가 제대로 걸린 것
print('저장 완료: filter_check.png')

# 참고: 이 실습에서 쓰는 데이터셋 저장소 이름 자체가 "palm-sEMG-doorknob-filtered"인 것에서
# 알 수 있듯, 원본 CSV가 이미 어느 정도 필터링되어 배포된 상태다. 그래서 필터 전(before)
# 그래프에도 60Hz 근처에 완전한 봉우리가 아니라 이미 옅은 골(dip)이 보일 수 있는데,
# 이는 우리 필터 코드가 잘못된 게 아니라 원본 데이터 특성이다. 그래도 위 수치 비교에서
# after < before 가 성립하면 우리가 짠 필터가 정상적으로 한 번 더 걸러낸 것이다.
