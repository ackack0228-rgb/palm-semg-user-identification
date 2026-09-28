# 2주차 실습 · 단계 4 : min-max 정규화
# 목적: 윈도우 조각마다 값의 범위를 0~1 사이로 맞춘다. 1주차에서 확인했듯 피험자마다
#       신호의 절대적인 크기(스케일)가 크게 다른데(A는 ±4, C는 ±0.3 등), 정규화를
#       하지 않으면 모델이 "누구인지"가 아니라 "신호가 얼마나 큰지"만 보고 구분해버릴
#       위험이 있다. 정규화로 채널·피험자 간 크기 차이를 없애준다.
# 실행 방법: semg-auth\week2 폴더 안에서 step1_filter.py, step3_window.py를 먼저 실행한 뒤
#   ../.venv/Scripts/python step4_normalize.py

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def minmax(w, eps=1e-8):
    """w: (윈도우수, 시간, 채널) -> 각 윈도우를 독립적으로 0~1 범위로 스케일링.

    axis=(1, 2)로 min/max를 구한다는 것은 "시간축과 채널축을 합쳐서" 즉 윈도우 하나
    전체(300개 시간 x 2채널 값 전부)를 통틀어 최솟값/최댓값을 구한다는 뜻이다.
    keepdims=True를 줘야 (윈도우수, 1, 1) 모양이 유지되어, 아래 뺄셈/나눗셈에서
    브로드캐스팅(broadcasting)으로 각 윈도우에 자기 자신의 min/max가 적용된다.
    """
    mn = w.min(axis=(1, 2), keepdims=True)
    mx = w.max(axis=(1, 2), keepdims=True)
    # eps(아주 작은 값)를 분모에 더해주는 이유: 만약 어떤 윈도우의 값이 전부 똑같다면
    # (mx - mn)이 0이 되어 0으로 나누는 오류(divide by zero)가 나기 때문에 안전장치로 넣는다.
    return (w - mn) / (mx - mn + eps)


if __name__ == '__main__':
    w = np.load(os.path.join(BASE_DIR, 'step3_w.npy'))
    wn = minmax(w)
    print('값 범위:', wn.min(), '~', wn.max())  # 0.0 ~ 1.0 근처가 나와야 정상
    np.save(os.path.join(BASE_DIR, 'step4_wn.npy'), wn)
    print('저장 완료: step4_wn.npy')

    # 참고(슬라이드 확인 방법 중 하나): axis=(1,2)는 "윈도우 단위" 정규화다.
    # 만약 채널별로 따로 정규화하고 싶으면 axis=1로 바꾸면 된다(각 채널이 독립적으로 0~1이 됨).
    # 이번 실습은 슬라이드 예시 그대로 axis=(1,2)를 사용했다.
