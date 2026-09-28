10-fold 교차검증 (시행 단위, 모든 모델 같은 fold)

| Model | Accuracy 평균 ± 표준편차 | Weighted F1 평균 ± 표준편차 | 완료 fold |
|---|---|---|---|
| 1D CNN | 92.97 ± 1.39% | 92.76 ± 1.46% | 10/10 |
| 2D CNN | 58.08 ± 1.82% | 55.95 ± 2.08% | 10/10 |
| ResNet18 | 88.55 ± 2.23% | 88.42 ± 2.24% | 10/10 |
| DenseNet161 | 90.53 ± 1.57% | 90.49 ± 1.58% | 10/10 |

Wilcoxon 부호순위 검정 (DenseNet161 vs 베이스라인, 유의수준 0.05)

| 비교 | 지표 | DenseNet161 승 / fold | 평균 차이 | 양측 p | 단측 p | 유의 | 더 높은 쪽 |
|---|---|---|---|---|---|---|---|
| DenseNet161 vs 1D CNN | accuracy | 0 / 10 | -2.44%p | 0.0020 | 1.0000 | O | 1D CNN |
| DenseNet161 vs 1D CNN | weighted_f1 | 0 / 10 | -2.28%p | 0.0020 | 1.0000 | O | 1D CNN |
| DenseNet161 vs 2D CNN | accuracy | 10 / 10 | +32.44%p | 0.0020 | 0.0010 | O | DenseNet161 |
| DenseNet161 vs 2D CNN | weighted_f1 | 10 / 10 | +34.53%p | 0.0020 | 0.0010 | O | DenseNet161 |
| DenseNet161 vs ResNet18 | accuracy | 8 / 10 | +1.98%p | 0.0195 | 0.0098 | O | DenseNet161 |
| DenseNet161 vs ResNet18 | weighted_f1 | 9 / 10 | +2.06%p | 0.0137 | 0.0068 | O | DenseNet161 |
