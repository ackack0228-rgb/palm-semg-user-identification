# 1주차 실습 · 단계 3 : 데이터 구조 확인
# 목적: data/ 폴더에 있는 CSV들이 몇 개인지, 배열 모양이 어떤지, 값 범위가 어떤지를
#       코드를 본격적으로 작성하기 전에 먼저 눈으로 확인한다.
#
# 실행 방법 (반드시 이 스크립트가 들어있는 week1 폴더에서 실행할 것):
#   ..\.venv\Scripts\python step3_explore.py   (PowerShell/명령 프롬프트, semg-auth\week1 안에서)
#   ../.venv/Scripts/python step3_explore.py   (Git Bash 등, semg-auth/week1 안에서)
# 다른 폴더에서 실행해도 되도록 아래에서 스크립트 자신의 위치를 기준으로 경로를 계산한다.

import numpy as np, glob, os
from collections import Counter

# __file__ = 이 스크립트 파일의 경로. abspath로 절대경로화한 뒤 dirname으로 "폴더"만 남긴다.
# 이렇게 하면 이 스크립트를 어느 위치(현재 작업 디렉터리)에서 실행하든 항상 같은 결과가 나온다.
# (실제로 예전에 다른 폴더에서 실행했다가 FileNotFoundError가 났던 적이 있어서 이렇게 고정함)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 폴더는 week1이 아니라 semg-auth 바로 아래(../data)에 있다.
# .venv와 data는 1주차·2주차·... 모든 주차가 함께 쓰는 "공용" 폴더이기 때문에
# 주차별 폴더(week1, week2, ...) 안이 아니라 semg-auth 루트에 둔 것.
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')  # -> semg-auth/data

# data/ 안에 다시 data/ 가 한 번 더 있고(git clone 저장소 구조 그대로),
# 그 안에 피험자 폴더 A~E, 그 안에 시행별 csv 파일(a (1).csv, a (2).csv, ...)이 있다.
# glob 패턴 '*/*.csv' 가 "아무 폴더 하나 아래의 csv 파일 전부"를 의미한다.
files = sorted(glob.glob(os.path.join(DATA_DIR, 'data', '*', '*.csv'), recursive=True))
print('파일 개수:', len(files))  # 5명 x 50시행 = 250개가 나와야 정상

# 파일을 하나도 못 찾았으면 (경로가 틀렸거나 git clone을 안 한 상태) 여기서 바로 멈추고
# 어떤 경로를 확인했는지 알려준다. 이렇게 해야 "왜 안 되지?"를 스스로 디버깅할 수 있다.
if not files:
    raise SystemExit(f"data/data/*/*.csv 에서 파일을 찾지 못했습니다. 확인한 경로: {DATA_DIR}")
print('예시 경로:', files[0])

# CSV 파일 하나를 읽어서 배열 모양(shape)과 값 범위를 확인한다.
# delimiter=',' : 콤마로 구분된 값들, skiprows=1 : 첫 줄은 헤더(컬럼 이름)라서 건너뜀
x = np.loadtxt(files[0], delimiter=',', skiprows=1)
print('배열 모양:', x.shape)          # (3000, 2) 예상 -> 3초 * 1000Hz = 3000행, 2채널(APB, ADM)
print('값 범위:', x.min(), '~', x.max())  # 필터링은 됐지만 min-max 정규화는 안 된 원래 값

# 피험자별로 파일이 정말 50개씩 균등하게 있는지 세어본다.
# os.path.dirname(f) -> ".../data/data/A" 같은 경로, basename -> 그 마지막 폴더 이름("A")
subs = [os.path.basename(os.path.dirname(f)) for f in files]
print('피험자별 시행 수:', Counter(subs))  # {'A': 50, 'B': 50, 'C': 50, 'D': 50, 'E': 50} 이어야 정상
