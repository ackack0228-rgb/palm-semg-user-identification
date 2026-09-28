| Model | Accuracy | Precision | Recall | F1-score | Params | Train time | Inference / sample |
|---|---|---|---|---|---|---|---|
| 1D CNN | 94.42% | 94.85% | 94.42% | 94.29% | 9,589 | 63 s | 1.04 ms |
| 2D CNN | 56.32% | 56.23% | 56.32% | 54.79% | 19,717 | 59 s | 0.41 ms |
| ResNet18 | 88.84% | 88.76% | 88.84% | 88.76% | 11,179,077 | 229 s | 4.50 ms |
| CNN-LSTM-Attention | 88.84% | 89.31% | 88.84% | 88.87% | 243,078 | 116 s | 1.44 ms |
| ConvNeXt | 86.00% | 87.37% | 86.00% | 85.72% | 27,823,973 | 810 s | 9.16 ms |
| Tri-CCNN | 50.84% | 62.43% | 50.84% | 47.30% | 2,213 | 92 s | 1.64 ms |
| E2CNN | 89.16% | 89.26% | 89.16% | 89.14% | 80,437 | 91 s | 1.19 ms |
| CNN-BiLSTM | 91.16% | 91.47% | 91.16% | 91.12% | 1,478,021 | 365 s | 4.50 ms |
| DenseNet161 | 91.37% | 91.80% | 91.37% | 91.35% | 26,483,045 | 1400 s | 33.44 ms |
| DenseNet161 x6 ensemble | 92.74% | 93.04% | 92.74% | 92.70% | 158,898,270 | - | 214.34 ms |

### 1D CNN

max error pair: {'true': 'E', 'pred': 'A', 'count': 18}

```
              precision    recall  f1-score   support

           A      0.904     0.995     0.947       190
           B      0.954     0.989     0.972       190
           C      0.896     1.000     0.945       190
           D      0.994     0.947     0.970       190
           E      0.993     0.789     0.880       190

    accuracy                          0.944       950
   macro avg      0.949     0.944     0.943       950
weighted avg      0.949     0.944     0.943       950

```

### 2D CNN

max error pair: {'true': 'B', 'pred': 'C', 'count': 70}

```
              precision    recall  f1-score   support

           A      0.901     0.958     0.929       190
           B      0.478     0.347     0.402       190
           C      0.423     0.621     0.503       190
           D      0.519     0.637     0.572       190
           E      0.490     0.253     0.333       190

    accuracy                          0.563       950
   macro avg      0.562     0.563     0.548       950
weighted avg      0.562     0.563     0.548       950

```

### ResNet18

max error pair: {'true': 'C', 'pred': 'D', 'count': 17}

```
              precision    recall  f1-score   support

           A      0.945     1.000     0.972       190
           B      0.891     0.905     0.898       190
           C      0.853     0.853     0.853       190
           D      0.874     0.874     0.874       190
           E      0.875     0.811     0.842       190

    accuracy                          0.888       950
   macro avg      0.888     0.888     0.888       950
weighted avg      0.888     0.888     0.888       950

```

### CNN-LSTM-Attention

max error pair: {'true': 'C', 'pred': 'E', 'count': 18}

```
              precision    recall  f1-score   support

           A      0.945     0.995     0.969       190
           B      0.865     0.879     0.872       190
           C      0.975     0.816     0.888       190
           D      0.898     0.884     0.891       190
           E      0.782     0.868     0.823       190

    accuracy                          0.888       950
   macro avg      0.893     0.888     0.889       950
weighted avg      0.893     0.888     0.889       950

```

### ConvNeXt

max error pair: {'true': 'E', 'pred': 'B', 'count': 53}

```
              precision    recall  f1-score   support

           A      0.984     1.000     0.992       190
           B      0.710     0.942     0.810       190
           C      0.897     0.874     0.885       190
           D      0.856     0.874     0.865       190
           E      0.921     0.611     0.734       190

    accuracy                          0.860       950
   macro avg      0.874     0.860     0.857       950
weighted avg      0.874     0.860     0.857       950

```

### Tri-CCNN

max error pair: {'true': 'B', 'pred': 'C', 'count': 108}

```
              precision    recall  f1-score   support

           A      0.971     0.874     0.920       190
           B      0.403     0.316     0.354       190
           C      0.377     0.853     0.523       190
           D      0.438     0.426     0.432       190
           E      0.933     0.074     0.137       190

    accuracy                          0.508       950
   macro avg      0.624     0.508     0.473       950
weighted avg      0.624     0.508     0.473       950

```

### E2CNN

max error pair: {'true': 'C', 'pred': 'D', 'count': 18}

```
              precision    recall  f1-score   support

           A      0.969     1.000     0.984       190
           B      0.866     0.884     0.875       190
           C      0.935     0.837     0.883       190
           D      0.855     0.868     0.862       190
           E      0.838     0.868     0.853       190

    accuracy                          0.892       950
   macro avg      0.893     0.892     0.891       950
weighted avg      0.893     0.892     0.891       950

```

### CNN-BiLSTM

max error pair: {'true': 'E', 'pred': 'B', 'count': 26}

```
              precision    recall  f1-score   support

           A      0.979     0.989     0.984       190
           B      0.823     0.953     0.883       190
           C      0.930     0.911     0.920       190
           D      0.915     0.905     0.910       190
           E      0.927     0.800     0.859       190

    accuracy                          0.912       950
   macro avg      0.915     0.912     0.911       950
weighted avg      0.915     0.912     0.911       950

```

### DenseNet161

max error pair: {'true': 'E', 'pred': 'B', 'count': 27}

```
              precision    recall  f1-score   support

           A      0.995     0.995     0.995       190
           B      0.824     0.958     0.886       190
           C      0.921     0.916     0.918       190
           D      0.895     0.900     0.898       190
           E      0.956     0.800     0.871       190

    accuracy                          0.914       950
   macro avg      0.918     0.914     0.913       950
weighted avg      0.918     0.914     0.913       950

```

### DenseNet161 x6 ensemble

max error pair: {'true': 'E', 'pred': 'B', 'count': 26}

```
              precision    recall  f1-score   support

           A      0.995     0.995     0.995       190
           B      0.841     0.947     0.891       190
           C      0.938     0.953     0.945       190
           D      0.922     0.937     0.930       190
           E      0.956     0.805     0.874       190

    accuracy                          0.927       950
   macro avg      0.930     0.927     0.927       950
weighted avg      0.930     0.927     0.927       950

```

