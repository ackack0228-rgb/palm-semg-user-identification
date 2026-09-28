# 2주차 실습 · 단계 3 : 슬라이딩 윈도우로 자르기
# 목적: 3초(3000샘플)짜리 시행 하나를 300ms(300샘플)짜리 짧은 조각 여러 개로 잘라서
#       학습 샘플 수를 늘린다. 이렇게 겹치게(오버랩) 자르는 이유는 데이터가 원래 250개
#       (=시행 250개)뿐이라 그대로는 딥러닝 학습에 턱없이 부족하기 때문이다.
# 실행 방법: semg-auth\week2 폴더 안에서 step1_filter.py를 먼저 실행한 뒤
#   ../.venv/Scripts/python step3_window.py

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# WIN=300 : 윈도우(조각) 길이. 1000Hz 샘플링이므로 300샘플 = 300ms.
# HOP=150 : 다음 윈도우로 이동하는 간격. 300의 절반이므로 "50% 오버랩"을 의미한다.
#           (오버랩을 주면 조각 수가 거의 2배로 늘어나면서 조각 사이 경계에서
#            정보가 끊기는 것도 줄여준다.)
WIN, HOP = 300, 150  # 샘플 단위 (1000Hz -> 300ms, 150ms)


def make_windows(x, win=WIN, hop=HOP):
    """x: (시간, 채널) -> (윈도우수, win, 채널)
    예) x가 (3000, 2)면, 300샘플씩 150샘플 간격으로 겹쳐 자르면
        n = (3000-300)//150 + 1 = 19개 윈도우가 나와서 결과는 (19, 300, 2).
    """
    # 몇 개의 윈도우가 나올지 미리 계산한다.
    # (전체 길이 - 윈도우 길이) 를 hop으로 나누면 "몇 번 이동할 수 있는지"가 나오고,
    # 처음 윈도우 1개를 더해줘야 총 개수가 맞는다.
    n = (len(x) - win) // hop + 1
    # 리스트 컴프리헨션으로 i번째 윈도우(x[i*hop : i*hop+win])를 n개 만든 뒤
    # np.stack으로 하나의 3차원 배열로 합친다.
    return np.stack([x[i * hop: i * hop + win] for i in range(n)])


if __name__ == '__main__':
    # step1에서 필터링까지 끝낸 신호(xf)를 불러와서 윈도우로 자른다.
    # (필터 -> 윈도우 순서가 맞다. 반대로 하면 윈도우 경계에서 필터의 edge effect가
    #  더 크게 나타날 수 있다.)
    xf = np.load(os.path.join(BASE_DIR, 'step1_xf.npy'))
    w = make_windows(xf)
    print('윈도우 배열 모양:', w.shape)  # (19, 300, 2) 가 나와야 정상
    np.save(os.path.join(BASE_DIR, 'step3_w.npy'), w)
    print('저장 완료: step3_w.npy')
