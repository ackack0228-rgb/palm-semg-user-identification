# 2주차 실습 · 단계 1 : 필터 적용하기
# 목적: 원신호에서 (1) 60Hz 전원선 잡음과 (2) 20~500Hz 밖의 불필요한 성분을 제거한다.
#
# 이 파일은 두 가지 역할을 겸한다.
#   1) 단독 실행하면(=python step1_filter.py) 예시 파일 하나로 필터 효과를 확인하고
#      step1_x.npy(필터 전), step1_xf.npy(필터 후)를 저장해서 다음 단계(step2)가 쓸 수 있게 한다.
#   2) step6_dataset.py 등 다른 스크립트에서 `from step1_filter import preprocess` 형태로
#      preprocess() 함수만 가져다 쓸 수도 있다. 그래서 실행 코드를
#      `if __name__ == '__main__':` 안에 넣어서, import될 때는 이 부분이 실행되지 않도록 했다.
#
# 실행 방법: semg-auth\week2 폴더 안에서
#   ../.venv/Scripts/python step1_filter.py

import numpy as np, glob, os
from scipy.signal import butter, iirnotch, filtfilt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 데이터셋은 week2 폴더가 아니라 semg-auth 바로 아래(../data)에 있다.
# (.venv, data 폴더는 모든 주차가 공유하는 공용 자원이라 주차 폴더 밖에 둔다.)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
FS = 1000  # 샘플링 주파수 1000Hz (1초에 1000개 샘플). 필터 설계에 항상 필요한 값.


def preprocess(x, fs=FS):
    """(시간, 채널) 모양의 원신호 x에 노치+대역통과 필터를 순서대로 적용해서 반환한다."""
    # 1) 60 Hz 전원선 잡음 제거 (notch filter)
    #    iirnotch(중심주파수, Q값, 샘플링주파수) -> 필터 계수 (b, a) 반환.
    #    Q=30 정도면 60Hz 근처만 좁게 잘라내고 나머지 주파수는 거의 건드리지 않는다.
    bn, an = iirnotch(60, 30, fs)
    # filtfilt = 필터를 신호에 앞으로 한 번, 뒤로 한 번(zero-phase) 걸어서 위상 지연이
    # 생기지 않게 한다. 실시간 처리가 아니라 저장된 신호를 통째로 분석하는 "오프라인
    # 분석"이라 이렇게 양방향으로 걸어도 문제없고, 오히려 신호가 밀리지 않아서 더 좋다.
    # axis=0 : 시간축(행) 방향으로 필터를 적용한다는 뜻. 이걸 빼먹으면 채널축(열) 방향으로
    #          잘못 걸려서 결과가 완전히 이상해진다 (2주차 슬라이드 "자주 나는 오류"에도 나옴).
    x = filtfilt(bn, an, x, axis=0)

    # 2) 20~499 Hz 대역만 통과 (band-pass filter, 4차)
    #    butter(차수, [저역/나이퀴스트, 고역/나이퀴스트], btype='band')
    #    - "나이퀴스트 주파수"는 fs/2 (=500Hz). 필터 설계 시 주파수는 항상 이 값으로
    #      나눠서 0~1 사이의 정규화된 값으로 넣어야 한다 (scipy의 규칙).
    #    - 500이 아니라 499를 쓴 이유: 500은 나이퀴스트 한계값과 정확히 같아서
    #      "0 < Wn < 1"을 벗어나 오류가 난다. 그래서 딱 1Hz 낮춘 499를 상한으로 쓴다.
    #    - 차수 4는 "얼마나 가파르게 잘라내는가"를 뜻하며, 4차는 실무에서 무난한 기본값.
    b, a = butter(4, [20 / (fs / 2), 499 / (fs / 2)], btype='band')
    return filtfilt(b, a, x, axis=0)


if __name__ == '__main__':
    # data/data/<피험자>/<시행>.csv 형태의 모든 파일 경로를 찾는다.
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data', '*', '*.csv')))
    if not files:
        raise SystemExit(f"data/data/*/*.csv 에서 파일을 찾지 못했습니다. 확인한 경로: {DATA_DIR}")

    # 예시로 첫 번째 파일 하나만 읽어서 필터 전/후를 비교해본다.
    # (전체 250개를 다 필터링하는 건 6단계 데이터셋 생성에서 한다 — 여기서는 확인만.)
    x = np.loadtxt(files[0], delimiter=',', skiprows=1)
    xf = preprocess(x)
    print('파일:', os.path.relpath(files[0], DATA_DIR))
    # 표준편차(std)가 필터 후 "조금" 줄어드는 것이 정상이다 (전원선 잡음 등 일부 성분이
    # 빠졌기 때문). 너무 많이 줄면 필터 대역이 너무 좁거나 뭔가 잘못됐다는 신호다.
    print('전:', x.std(), ' 후:', xf.std())

    # 다음 단계(step2_verify.py)가 이어서 쓸 수 있도록 필터 전/후 신호를 파일로 저장.
    # "한 단계 끝날 때마다 결과를 파일로 저장해 둘 것"이라는 슬라이드 지침을 따른 것.
    np.save(os.path.join(BASE_DIR, 'step1_x.npy'), x)
    np.save(os.path.join(BASE_DIR, 'step1_xf.npy'), xf)
    print('저장 완료: step1_x.npy, step1_xf.npy')
