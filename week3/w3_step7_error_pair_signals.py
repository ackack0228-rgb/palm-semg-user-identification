# 3주차 실습 · 단계 2 보완 — 최대 오류 쌍(E → B)의 신호를 나란히 그려 원인 확인
#
# 3주차 슬라이드 10쪽: "그 두 사람의 신호를 나란히 그려 왜 그런지 확인해 볼 것".
# 모델(CNN)을 쓰지 않고 입력 신호 자체만 본다. 모델과 무관하게 신호에서도 E와 B가
# 파지 구간에서 가깝다면, E → B 오류는 모델 탓이 아니라 데이터의 성질이라는 근거가 된다.
#
# 1) figures/error_pair_signals.png : B·E 50시행의 RMS 포락선(시행마다 최댓값 1로 맞춤) 평균 ± 표준편차
# 2) figures/error_pair_cwt.png     : B·E의 구간별 평균 CWT 지도(모델 입력과 같은 전처리)
# 3) results/error_pair_analysis.md : 구간별 스펙트럼 프로파일로 최근접 중심(nearest centroid)
#    판별 — 학습 없는 가장 단순한 분류기로 "누구로 가장 가깝게 보이는지"를 센다(leave-one-out).
#
# 실행: week3 폴더에서  ../.venv/Scripts/python w3_step7_error_pair_signals.py   (CPU, 약 1분)

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common  # noqa: F401  (week2 경로를 sys.path에 넣는다)
from step1_filter import preprocess
from step3_window import make_windows
from step4_normalize import minmax
from step5_cwt import to_cwt
from step6_dataset import load, subject_of

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')
SUBJ = ['A', 'B', 'C', 'D', 'E']
FS = 1000

# 윈도우 위치(pos)는 시작 시각 pos*150 ms, 길이 300 ms. 한 구간 안에 완전히 들어가는 윈도우만 쓴다.
PHASES = {
    'grasp (0-1 s)': [0, 1, 2, 3, 4],             # 0~900 ms
    'rotation (1-2 s)': [7, 8, 9, 10, 11],        # 1050~1950 ms
    'stationary (2-3 s)': [14, 15, 16, 17, 18],   # 2100~3000 ms
}


def rms_envelope(sig, win=50):
    """50 ms 이동 RMS 포락선, 시행마다 채널별 최댓값을 1로 맞춘다(모델도 min-max로 진폭을 지운다)."""
    k = np.ones(win) / win
    env = np.stack([np.sqrt(np.convolve(sig[:, c] ** 2, k, mode='same')) for c in range(2)], 1)
    return env / env.max(0, keepdims=True)


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.csv')))
    env = {s: [] for s in SUBJ}
    cwt = {s: [] for s in SUBJ}   # 시행마다 (19, 3, 32, 300)
    for f in files:
        s = subject_of(f)
        sig = preprocess(load(f))
        env[s].append(rms_envelope(sig))
        cwt[s].append(np.stack([to_cwt(w) for w in minmax(make_windows(sig))]))
    env = {s: np.stack(v) for s, v in env.items()}   # (50, 3000, 2)
    cwt = {s: np.stack(v) for s, v in cwt.items()}   # (50, 19, 3, 32, 300)
    print('loaded', {s: cwt[s].shape for s in SUBJ})

    # ---- 1) 포락선 그림 ------------------------------------------------------
    t = np.arange(3000) / FS
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for c, name in enumerate(['ch1 APB', 'ch2 ADM']):
        ax = axes[c]
        for s, col in [('B', 'tab:blue'), ('E', 'tab:red')]:
            m, sd = env[s][:, :, c].mean(0), env[s][:, :, c].std(0)
            ax.plot(t, m, color=col, lw=1.2, label=f'{s} mean (50 trials)')
            ax.fill_between(t, m - sd, m + sd, color=col, alpha=0.15)
        for x in (1.0, 2.0):
            ax.axvline(x, color='k', ls='--', lw=0.8)
        ax.set_ylabel(f'{name}\nnormalized RMS')
        ax.legend(loc='upper right', fontsize=8)
    axes[0].set_title('B vs E: RMS envelope (mean ± std)  |  grasp / rotation / stationary')
    axes[1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'error_pair_signals.png'), dpi=120)
    plt.close(fig)

    # ---- 2) 구간별 평균 CWT 지도 (3번째 채널 = 두 채널 평균, 모델 입력과 같음) ----
    # 윈도우 양 끝(큰 스케일)에는 경계 효과가 크게 나타나 색 범위를 차지하므로,
    # 색 범위는 가운데 구간(60~240 ms)의 99 백분위수로 정한다. 3행은 E − B 차이.
    mean_map = {(s, p): cwt[s][:, PHASES[p], 2].mean((0, 1)) for s in ('B', 'E') for p in PHASES}
    vmax = np.percentile(np.stack([m[:, 60:240] for m in mean_map.values()]), 99)
    dmax = max(np.abs(mean_map[('E', p)] - mean_map[('B', p)])[:, 60:240].max() for p in PHASES)
    fig, axes = plt.subplots(3, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    for j, p in enumerate(PHASES):
        for i, s in enumerate(['B', 'E']):
            im = axes[i, j].imshow(mean_map[(s, p)], aspect='auto', origin='lower', cmap='viridis',
                                   vmin=0, vmax=vmax, extent=[0, 300, 1, 32])
            axes[i, j].set_title(f'{s} · {p}', fontsize=10)
        imd = axes[2, j].imshow(mean_map[('E', p)] - mean_map[('B', p)], aspect='auto', origin='lower',
                                cmap='RdBu_r', vmin=-dmax, vmax=dmax, extent=[0, 300, 1, 32])
        axes[2, j].set_title(f'E − B · {p}', fontsize=10)
        axes[2, j].set_xlabel('time in window (ms)')
    for i in range(3):
        axes[i, 0].set_ylabel('CWT scale')
    fig.colorbar(im, ax=axes[:2], shrink=0.8, label='|CWT| (mean, clipped)')
    fig.colorbar(imd, ax=axes[2], shrink=0.8, label='E − B')
    fig.suptitle('Mean CWT input (channel 3 = avg of APB, ADM), B vs E')
    fig.savefig(os.path.join(FIG_DIR, 'error_pair_cwt.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ---- 3) 학습 없는 최근접 중심 판별 (leave-one-trial-out) ---------------------
    # 시행 하나의 구간 특징 = 그 구간 윈도우들의 CWT를 시간축으로 평균한 스케일 프로파일 (3채널 x 32 = 96차원).
    # 윈도우 안의 시간 위치는 시행마다 어긋나므로 평균으로 지우고, "어느 주파수가 얼마나 강한가"만 남긴다.
    out = {}
    lines = ['# 최대 오류 쌍(E → B) 신호 분석 — 학습 없는 최근접 중심 판별', '',
             '시행마다 구간 윈도우들의 CWT를 시간축으로 평균한 96차원 스펙트럼 프로파일을 만들고, '
             '각 시행을 "자기 자신을 뺀" 피험자별 평균(중심)과 비교해 가장 가까운 사람으로 판정했다 '
             '(유클리드 거리, 250시행 전체, leave-one-trial-out). CNN을 전혀 쓰지 않은 기준선이다.', '']
    for p, pos in PHASES.items():
        feat = {s: cwt[s][:, pos].mean(axis=(1, 4)).reshape(50, -1) for s in SUBJ}   # (50, 96)
        sums = {s: feat[s].sum(0) for s in SUBJ}
        conf = np.zeros((5, 5), int)
        for i, s in enumerate(SUBJ):
            for k in range(50):
                d = []
                for j, s2 in enumerate(SUBJ):
                    cen = (sums[s2] - feat[s][k]) / 49 if s2 == s else sums[s2] / 50
                    d.append(np.linalg.norm(feat[s][k] - cen))
                conf[i, int(np.argmin(d))] += 1
        acc = np.trace(conf) / conf.sum()
        out[p] = {'accuracy': float(acc), 'confusion': conf.tolist(),
                  'per_subject': {s: int(conf[i, i]) for i, s in enumerate(SUBJ)}}
        lines += [f'## {p} — 전체 {acc*100:.1f}% ({np.trace(conf)}/250)', '',
                  '| 실제 \\ 판정 | ' + ' | '.join(SUBJ) + ' | 정답률 |',
                  '|---|' + '---|' * 6]
        for i, s in enumerate(SUBJ):
            lines.append(f'| {s} | ' + ' | '.join(str(v) for v in conf[i]) + f' | {conf[i, i]*2}% |')
        lines.append('')
        print(p, f'{acc*100:.1f}%', 'E row:', conf[4].tolist())

    with open(os.path.join(RES_DIR, 'error_pair_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RES_DIR, 'error_pair_analysis.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('saved figures/error_pair_signals.png, figures/error_pair_cwt.png, results/error_pair_analysis.md')


if __name__ == '__main__':
    main()
