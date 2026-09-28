# 1주차 실습 · 단계 4 : 신호 시각화
# 목적: sEMG 원신호를 실제로 그려봐서 "파지(0~1s) - 회전(1~2s) - 정지(2~3s)" 3구간이
#       눈으로 보이는지, 피험자마다 신호 크기(스케일)가 얼마나 다른지 확인한다.
#
# 실행 방법: semg-auth\week1 폴더 안에서 실행
#   ../.venv/Scripts/python step4_plot.py
# 결과: signal_example.png, signal_compare_subjects.png 두 그림이 이 폴더에 저장된다.

import numpy as np, glob, os
import matplotlib
# 'Agg' 백엔드 = 화면에 창을 띄우지 않고 파일로만 저장하는 모드.
# 서버/터미널 환경(디스플레이 없음)에서 matplotlib을 쓸 때 필수로 지정해야 오류가 안 난다.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# step3_explore.py와 동일한 방식으로 "스크립트 자신의 위치" 기준 절대경로를 만든다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')  # semg-auth/data (주차 폴더 공용)
files = sorted(glob.glob(os.path.join(DATA_DIR, 'data', '*', '*.csv'), recursive=True))
if not files:
    raise SystemExit(f"data/data/*/*.csv 에서 파일을 찾지 못했습니다. 확인한 경로: {DATA_DIR}")

# ── 그림 1: 피험자 A의 첫 번째 시행 신호 (2채널을 위아래로 배치) ──────────────
x = np.loadtxt(files[0], delimiter=',', skiprows=1)
# 혹시 파일이 (채널, 시간) 순서로 저장돼 있으면 (시간, 채널)로 뒤집어준다.
# 우리 데이터는 보통 (3000, 2)라서 이 조건은 대부분 실행되지 않지만, 다른 파일 형식이
# 섞여도 안전하게 동작하도록 방어적으로 넣어둔 코드.
if x.shape[0] < x.shape[1]:
    x = x.T
# 샘플링 레이트가 1000Hz이므로, 인덱스를 1000으로 나누면 "초" 단위 시간축이 된다.
t = np.arange(len(x)) / 1000.0

fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
for ch in range(2):
    ax[ch].plot(t, x[:, ch], lw=0.6)
    ax[ch].set_ylabel(f'ch{ch+1}')  # ch1=APB(무지외전근), ch2=ADM(소지외전근)
for a in ax:
    # 1.0초, 2.0초 지점에 빨간 점선을 그어 파지/회전/정지 구간 경계를 표시한다.
    a.axvline(1.0, color='r', ls='--')
    a.axvline(2.0, color='r', ls='--')
ax[1].set_xlabel('time (s)')
# 제목에 상대경로를 넣어서 이 그림이 어떤 파일에서 나온 건지 나중에도 알 수 있게 한다.
ax[0].set_title(os.path.relpath(files[0], BASE_DIR))
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'signal_example.png'), dpi=120)
plt.close(fig)  # 다음 그림을 그리기 전에 반드시 닫아서 메모리/축이 섞이지 않게 한다.

# ── 그림 2: 피험자 A·C·E 비교 (스케일 차이를 보여주기 위한 선택) ────────────
# A, C, E 세 명만 고른 이유: 관찰 결과 A는 신호 크기가 크고(±4), C는 작아서(±0.3)
# "피험자마다 근전도 신호의 절대적 크기가 크게 다르다"는 점을 대비시켜 보여주기 좋았기 때문.
subjects = ['A', 'C', 'E']
fig, ax = plt.subplots(len(subjects), 2, figsize=(10, 6), sharex=True)
for i, s in enumerate(subjects):
    # 각 피험자 폴더에서 정렬 후 첫 번째 파일(=1번째 시행)만 대표로 사용
    path = sorted(glob.glob(os.path.join(DATA_DIR, 'data', s, '*.csv')))[0]
    xs = np.loadtxt(path, delimiter=',', skiprows=1)
    if xs.shape[0] < xs.shape[1]:
        xs = xs.T
    ts = np.arange(len(xs)) / 1000.0
    for ch in range(2):
        ax[i, ch].plot(ts, xs[:, ch], lw=0.6)
        ax[i, ch].axvline(1.0, color='r', ls='--')
        ax[i, ch].axvline(2.0, color='r', ls='--')
        ax[i, ch].set_ylabel(f'{s}\nch{ch+1}')
ax[-1, 0].set_xlabel('time (s)')
ax[-1, 1].set_xlabel('time (s)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'signal_compare_subjects.png'), dpi=120)
plt.close(fig)

print('저장 완료: signal_example.png, signal_compare_subjects.png')
