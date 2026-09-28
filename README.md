# 손바닥 sEMG 기반 문손잡이 사용자 식별 — 논문 재현 및 모델 성능 비교

대구가톨릭대학교 「딥러닝프로그래밍」 Step 1 과제 (21113103 김근영)

- 재현 대상 논문: Shin, Y., Kim, J., & Choi, S.-I. (2026). *Palm sEMG-based user identification during doorknob rotation using a convolutional neural network.* Scientific Reports, 16, 22244.
- 데이터: [sea3551/palm-sEMG-doorknob-filtered](https://github.com/sea3551/palm-sEMG-doorknob-filtered)

## 한눈에 보는 결과

| 항목 | 논문 | 본 재현 |
|---|---|---|
| 테스트 정확도 (DenseNet161, 단일 모델) | 94.00% | **91.37%** |
| macro F1 | 93.99% | **91.35%** |
| 5-fold 교차검증 (정확도) | 91.66 ± 2.78% | **CV_ACC** |
| 최대 오류 쌍 | D → C (13건) | **E → B (27건)** |
| 파라미터 수 | 26,483,045 | 26,483,045 (동일) |
| 학습 시간 (45 epoch) | 6,441 s | 1,400 s (RTX 3060 Laptop, 환경이 달라 직접 비교 불가) |
| (참고) DenseNet161 6개 앙상블 | – | 92.74% |

---

## 1. 코드 설명

### 1.1 문제 정의

문손잡이를 잡고 돌릴 때 손바닥 근육(APB·ADM)에서 측정한 2채널 sEMG 신호만으로 **등록된 5명(A~E) 중 누구인지** 맞히는 **closed-set 사용자 식별** 문제다. 입력은 300 ms 신호 조각 하나, 출력은 5개 클래스 중 하나다. 성공 기준은 논문의 테스트 정확도 94.00%와 5-fold 교차검증 91.66 ± 2.78%이다.

### 1.2 사용한 데이터 및 전처리

- **규모**: 피험자 5명 × 50시행 = 250개 CSV. 시행 하나는 3초 × 1000 Hz × 2채널(3000 × 2)이다.
- **분할**: **시행(파일) 단위로 먼저** 8:2 분할한다(stratify, `random_state=42`). 학습 200시행, 테스트 50시행(피험자당 10시행)이며, 학습과 테스트에 겹치는 시행이 0건임을 코드로 확인한다(`week2/step6_dataset.py`).
  윈도우를 먼저 만들고 나누면 같은 시행에서 150 ms만 어긋난 거의 같은 조각이 학습과 테스트 양쪽에 들어가 누수가 생기므로, 반드시 이 순서를 지킨다.
- **전처리 순서** (각 시행 안에서만 수행):
  1. 60 Hz notch(Q=30) → 20~499 Hz 4차 Butterworth band-pass(`filtfilt`)
  2. 300 ms 윈도우, 150 ms hop(50% overlap) → 시행당 19개 윈도우
  3. 윈도우별 min-max 정규화(0~1)
  4. CWT(Morlet, 스케일 1~32) → |계수|를 ch1, ch2, (ch1+ch2)/2의 3채널로 쌓아 **3 × 32 × 300** 텐서
- 학습 3,800개 / 테스트 950개 윈도우이며, 클래스별 개수는 760/190개로 완전히 균형이다.

### 1.3 사용한 모델

| 모델 | 구조 | 파라미터 |
|---|---|---|
| **DenseNet161** (제안, 논문 모델) | Dense block [6, 12, 36, 24], growth rate 48, 분류기 Dropout(0.5)+Linear(2208→5) | 26,483,045 |
| ResNet18 (베이스라인) | torchvision `resnet18`, 분류기 Dropout(0.5)+Linear(512→5) | 11,179,077 |
| 2D CNN (베이스라인) | Conv(3→32)-ReLU-MaxPool-Conv(32→64)-ReLU-GAP-Dropout-Linear(64→5) (3주차 슬라이드 구조) | 19,717 |

세 모델 모두 사전학습 가중치 없이(`weights=None`) 처음부터 학습한다. 논문도 ImageNet 사전학습을 쓰지 않았다.

### 1.4 학습 및 테스트 방법

모든 모델이 **완전히 같은 조건**으로 학습되도록 `week3/common.py`의 `train_model()` 하나로 통일했다.

| 설정 | 값 |
|---|---|
| 데이터 분할 | 동일 (`step6_dataset.npz`, 시행 단위 8:2, seed 42) |
| Epoch / Batch | 45 / 16 |
| Optimizer | Adam, lr = 1e-3, weight decay = 1e-4 |
| LR scheduler | CosineAnnealingLR (T_max = 45) |
| 증강 | 가우시안 노이즈(σ=0.02) + 시간축(≤30)/스케일축(≤4) 랜덤 마스킹, 각 50% |
| Seed | 1 (`torch`, `random`, `numpy`, DataLoader shuffle까지 고정) |

- 테스트: 950개 윈도우에 대해 Accuracy, macro Precision/Recall/F1, 혼동행렬을 구한다(`model.eval()`, `torch.no_grad()`).
- 5-fold 교차검증: 250개 **시행 파일**을 `StratifiedKFold(5, shuffle=True, random_state=42)`로 나누고, 폴드마다 윈도우와 CWT를 새로 만든 뒤 **새 모델**을 학습한다.
- 계산 비용: 파라미터 수, 45 epoch 학습 시간, 샘플 1개 추론 시간(예열 10회 후 100회 평균, `torch.cuda.synchronize()`)을 측정한다.

> 논문 설정에 없는 weight decay, cosine scheduler, 증강은 2주차에 45 epoch 단일 학습(88.21%)이 과적합되는 것을 확인하고 추가했다. 비교의 공정성을 위해 베이스라인에도 똑같이 적용했다.

### 1.5 실행 방법

```bash
# 0) 환경 (Python 3.14, CUDA 12.8 GPU 권장 — CPU도 동작하지만 약 10배 느림)
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU 사용 시

# 1) 데이터 받기 (저장소 루트에 data/ 로)
git clone https://github.com/sea3551/palm-sEMG-doorknob-filtered.git data

# 2) 데이터셋 생성: 시행 단위 분할 → 필터 → 윈도우 → 정규화 → CWT  (약 30초, 약 500MB)
cd week2
python step6_dataset.py

# 3) 제안 모델 DenseNet161 학습 (45 epoch, RTX 3060 Laptop 기준 약 23분)
python -u step13_train_seed.py 1

# 4) 베이스라인 학습 (같은 조건)
cd ../week3
python -u w3_step3_baselines.py cnn2d        # 약 1분
python -u w3_step3_baselines.py resnet18     # 약 4분

# 5) 평가: 지표 · 혼동행렬 · 계산 비용 → results/, figures/
python w3_step1_evaluate.py

# 6) 5-fold 교차검증 (약 2시간, 중간에 끊겨도 다시 실행하면 이어서 진행)
python -u w3_step4_cv.py
```

`w3_step1_evaluate.py`는 참고용 6모델 앙상블도 함께 평가한다. 이를 위해서는 2주차 체크포인트 6개가 필요하다: `step7_train.py` → `step10_train_45epoch.py`, `step11_train_improved.py`, `step13_train_seed.py 1/2/3`, `step16_dataset_scale64.py` → `step17_train_scale64.py`.

### 1.6 코드 파일 설명

| 파일 | 설명 |
|---|---|
| `week1/step3_explore.py`, `step4_plot.py` | 데이터 구조 파악, 원신호 시각화 |
| `week2/step1_filter.py` | 60 Hz notch + 20~499 Hz band-pass (`preprocess`) |
| `week2/step2_verify.py` | 필터 전후 스펙트럼 비교 → `filter_check.png` |
| `week2/step3_window.py` | 300 ms / 150 ms hop 슬라이딩 윈도우 (`make_windows`) |
| `week2/step4_normalize.py` | 윈도우별 min-max 정규화 (`minmax`) |
| `week2/step5_cwt.py` | Morlet CWT → 3×32×300 텐서 (`to_cwt`) |
| `week2/step6_dataset.py` | **시행 단위 분할 후** 데이터셋 생성 (누수 방지, `build`) |
| `week2/step7_train.py` | DenseNet161 5 epoch 기본 학습 (2주차 과제) |
| `week2/step8_evaluate.py` | 체크포인트 테스트 정확도 평가 |
| `week2/step10`~`step13` | 45 epoch 학습, 개선 레시피, 다중 seed 학습 |
| `week2/step14`~`step21` | 앙상블, 혼동행렬, CWT 스케일 64 실험, 오류 위치 진단 |
| `week3/common.py` | **공통 학습 레시피**, 모델 정의(`build_model`), 파라미터 수, 예측 |
| `week3/w3_step1_evaluate.py` | 지표 · 혼동행렬 그림 · 최대 오류 쌍 · 추론 시간 |
| `week3/w3_step3_baselines.py` | 베이스라인(2D CNN, ResNet18) 학습 |
| `week3/w3_step4_cv.py` | 시행 단위 5-fold 교차검증 |
| `week3/results/` | `metrics.json`, `metrics_table.md`, `cv_folds.json`, `train_*.json` |
| `week3/figures/` | `confusion.png`(제안 모델), `confusion_<model>.png` |

---

## 2. 모델 성능 비교

같은 테스트셋(50시행 = 950 윈도우)에서의 결과. Precision, Recall, F1은 macro 평균이다.

METRICS_TABLE

5-fold 교차검증 (DenseNet161, 시행 단위 StratifiedKFold):

CV_TABLE

### 성능 분석

- **DenseNet161이 단일 모델 중 가장 높았다** (정확도 91.37%, F1 91.35%). ResNet18(88.84%)보다 2.5%p, 2D CNN(56.32%)보다 35%p 높다. 정확도와 F1이 거의 같아 특정 클래스로 예측이 쏠리지는 않았다.
- **이유**: 입력은 채널 2개, 300 ms라 정보량이 적다. DenseNet은 앞 층의 특징 맵을 뒤 층에 모두 이어 붙이므로(dense connection) 얕은 층의 미세한 시간-주파수 패턴이 깊은 층까지 사라지지 않고 전달된다. ResNet18도 잔차 연결로 이를 어느 정도 해내지만, 층이 얕아 E·C·D처럼 비슷한 사람들 사이의 차이를 덜 잡아냈다.
- **2D CNN은 과소적합**이었다. 45 epoch 뒤에도 학습 loss가 약 1.05에 머물렀다(ResNet18 0.04, DenseNet161 0.09). 합성곱이 2층뿐이고 곧바로 전역 평균 풀링을 하므로, "언제 어느 주파수가 강했는지"라는 위치 정보가 사라진다. 그래서 신호 모양이 뚜렷이 다른 A(95.8%)만 잘 맞히고 나머지는 25~64%에 그쳤다.
- **비용 대비 성능**: ResNet18은 DenseNet161보다 파라미터가 42%, 학습 시간이 16%, 추론 시간이 약 1/8 수준인데 정확도 차이는 2.5%p다. 실시간 문손잡이 인증처럼 임베디드 환경을 생각하면 ResNet18도 현실적인 선택지다.
- **참고: 6모델 앙상블**(DenseNet161 스케일32 5개 + 스케일64 1개)은 92.74%로 단일 모델보다 1.4%p 높다. 그러나 조건이 달라(여러 seed와 설정, 6배 비용) 위 공정 비교표에는 넣지 않고 참고로만 적었다.

---

## 3. Confusion Matrix 분석

행 = 실제(true), 열 = 예측(predicted). 빨간 테두리는 최대 오류 쌍이다.

### DenseNet161 (제안 모델)

![DenseNet161](week3/figures/confusion.png)

- **가장 잘 분류된 클래스**: A (189/190, 99.5%)
- **가장 많이 오분류된 클래스**: E (재현율 80.0%, 38개 오류)
- **주요 오분류 유형**: **E → B 27건** (최대 오류 쌍). 그다음은 C → D 11건, D → C 9건, D → B 8건이다.
- **오분류가 발생한 이유**:
  - E의 오류 38건 중 27건(71%)이 B로 몰린다. 2주차에 시드와 설정을 바꾼 5개 모델 앙상블에서도 똑같이 E → B가 최대였다. 그러므로 초기화 운이 아니라 **입력 표현 자체에서 E와 B가 비슷하게 보이는 것**이다.
  - 오류를 **시행 안 위치별로** 나눠 보면(`week2/step21_window_position_diagnosis.py`, 6모델 앙상블 예측 기준), E는 **파지 구간(0~1 s) 윈도우 정확도가 56%**(pos 1~4는 40~60%)로 떨어진다. 반면 회전 구간(1~2 s)은 88%, 정지 구간(2~3 s)은 92%다. 다른 4명은 파지 구간에서도 96~100%였다. 즉 E는 손잡이를 **잡는 순간**의 근활성 패턴이 시행마다 일정하지 않거나 B와 겹친다. 문을 실제로 **돌리는** 구간에서는 E도 잘 구분된다.
  - C ↔ D 혼동(11 + 9건)은 논문의 최대 오류 쌍(D → C 13건)과 같은 쌍이다. 두 사람의 근육 사용 방식이 원래 비슷하다는 점이 재현에서도 드러난다.

### ResNet18

![ResNet18](week3/figures/confusion_resnet18.png)

- **가장 잘 분류된 클래스**: A (190/190, 100%)
- **가장 많이 오분류된 클래스**: E (재현율 81.1%)
- **주요 오분류 유형**: **C → D 17건** (최대), E → B 17건, D → C 16건
- **오분류가 발생한 이유**: C ↔ D 양방향 혼동이 33건으로 가장 크다. 논문에서 지적한 D → C 쌍과 같다. 오류가 한 쌍에 몰리지 않고 E → B, E → A, B → E 등으로 **퍼져 있다**. 모델 용량이 작아 특정 쌍 하나를 못 가르는 것이 아니라 전반적으로 경계가 덜 날카롭다는 뜻이다.

### 2D CNN

![2D CNN](week3/figures/confusion_cnn2d.png)

- **가장 잘 분류된 클래스**: A (182/190, 95.8%)
- **가장 많이 오분류된 클래스**: E (재현율 25.3%), B (34.7%)
- **주요 오분류 유형**: **B → C 70건** (최대), C → D 45건, D → C 47건, E는 B·C·D로 고르게 흩어진다(44/44/47건)
- **오분류가 발생한 이유**: 과소적합이다. 전역 평균 풀링으로 시간 정보가 사라져 "평균적인 주파수 분포"만 남는다. 피험자 간 차이가 큰 A만 구분되고 B~E는 사실상 뭉뚱그려진다. E는 거의 무작위로 흩어지는데, 이는 E가 가진 구분 단서(회전 구간의 패턴)가 시간 위치에 의존한다는 위 분석과도 맞는다.

### (참고) DenseNet161 6모델 앙상블

![Ensemble](week3/figures/confusion_ensemble6.png)

- C ↔ D 오류가 20건에서 10건으로 절반이 됐다. 그러나 **E → B는 27건에서 26건으로 그대로**였다. 앙상블은 모델마다 다르게 틀리는 "변동성 오류"는 줄이지만, 모든 모델이 똑같이 틀리는 E → B 같은 "구조적 오류"는 줄이지 못한다.

---

## 4. 최종 결과

- **가장 성능이 좋은 모델**: DenseNet161 (단일 모델 91.37% / F1 91.35%, 참고로 6모델 앙상블 92.74%)
- **가장 성능이 낮은 모델**: 2D CNN (56.32% / F1 54.79%, 과소적합)
- **주요 오분류 클래스**: E. 특히 **E → B**(DenseNet161 27건)이며, 원인은 E의 파지 구간(0~1 s) 신호다. 부차적으로 C ↔ D 혼동이 있는데, 이는 논문과 같은 쌍이다.

### 논문과 다르게 나온 이유

단일 split 테스트 정확도는 91.37%로 논문(94.00%)보다 2.6%p 낮다. 5-fold 평균은 CV_ACC로 논문(91.66 ± 2.78%)과 CV_COMPARE. 차이의 가장 큰 원인은 **테스트셋이 작다**는 점이라고 본다. 피험자당 테스트 시행이 10개뿐이므로 **시행 하나(19윈도우)를 통째로 틀리면 그 피험자 정확도가 10%p 흔들린다**. 실제로 같은 레시피에서 seed만 바꿔도 테스트 정확도가 90.11~91.37%로 달라졌고, 2주차 초기 설정 두 개(88.21%, 89.05%)는 D 피험자 정확도가 93.7% ↔ 87.9%로 뒤집혔다. 논문의 분할(어떤 시행이 테스트에 들어갔는지)은 공개되지 않았으므로 94.00%라는 단일 수치를 그대로 맞히는 것은 기대하기 어렵다. 대신 교차검증 평균과 표준편차로 비교하는 것이 더 공정하다. 최대 오류 쌍이 논문(D → C)과 다르게 E → B로 나온 것도 같은 이유로, 어느 E 시행이 테스트에 들어갔는지에 따라 달라질 수 있다. 다만 C ↔ D 혼동은 여기서도 두 번째로 큰 오류로 나타났다.

### 한계

1. **피험자 5명의 closed-set 문제다.** 모델은 "등록된 5명 중 누구와 가장 비슷한가"만 답하므로, 등록되지 않은 사람이 문을 잡아도 반드시 5명 중 한 명으로 판정한다. 실제 출입 인증에 쓰려면 미등록자를 거부하는 open-set 판정(예: 확신도 임계값)과 더 많은 사용자에 대한 검증이 필요하다.
2. **단일 세션 데이터다.** 50시행이 같은 날, 같은 전극 부착 상태에서 기록되어 학습과 테스트 시행이 매우 가까운 조건을 공유한다. 전극을 다시 붙이거나 며칠이 지나 피부 상태·근피로가 달라지면 신호가 바뀌므로, 세션 간 일반화 성능은 이 결과보다 낮을 가능성이 크다.
3. **실험실 조건이다.** 정해진 손잡이를 정해진 동작(파지-회전-정지 각 1초)으로 돌렸다. 실제 환경에서는 문 종류, 잡는 손, 짐을 든 상태, 서두름 등으로 동작 속도와 힘이 달라진다. E의 파지 구간처럼 동작이 조금만 흔들려도 정확도가 크게 떨어진 것을 보면 이 변화에 민감할 수 있다.
4. **테스트셋이 작고, 개선 과정에서 반복 사용했다.** 피험자당 테스트 시행이 10개라 시행 하나가 10%p를 좌우한다. 또 2주차의 개선(레시피 변경, 앙상블 구성)을 이 같은 테스트셋의 정확도를 보며 골랐으므로, 앙상블 92.74%에는 테스트셋에 맞춰진 낙관적 편향이 들어 있을 수 있다. 이 때문에 5-fold 교차검증 결과를 함께 보고한다.
5. **논문 설정과 일부 다르다.** weight decay, cosine scheduler, 데이터 증강은 논문에 없는 추가 기법이다. Dropout 비율(0.5)도 논문에 수치가 명시되지 않아 임의로 정했다. 따라서 이 결과는 논문 방법의 엄밀한 재현이라기보다 "같은 파이프라인에 학습 기법을 보강한 결과"로 봐야 한다.

---

## 5. 재현 정보

| 항목 | 값 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) |
| Python | 3.14.0 |
| 주요 패키지 | torch 2.11.0+cu128, torchvision 0.26.0, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, PyWavelets 1.10.0 |
| Seed | 데이터 분할 42, 교차검증 분할 42, 학습 1 (앙상블 구성 모델은 2, 3도 사용) |
| 실행 순서 | 위 1.5절 |

모델 가중치(`.pt`, 1개당 최대 103 MB)와 CWT 데이터셋(`.npz`, 약 500 MB)은 GitHub 용량 제한 때문에 저장소에 넣지 않았다. 위 실행 순서로 다시 만들 수 있다.
