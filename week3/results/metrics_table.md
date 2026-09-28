| Model | Accuracy | Precision | Recall | F1-score | Params | Train time | Inference / sample |
|---|---|---|---|---|---|---|---|
| 2D CNN | 56.32% | 56.23% | 56.32% | 54.79% | 19,717 | 59 s | 0.41 ms |
| ResNet18 | 88.84% | 88.76% | 88.84% | 88.76% | 11,179,077 | 229 s | 4.08 ms |
| DenseNet161 | 91.37% | 91.80% | 91.37% | 91.35% | 26,483,045 | 1400 s | 34.12 ms |
| DenseNet161 x6 ensemble | 92.74% | 93.04% | 92.74% | 92.70% | 158,898,270 | - | 209.70 ms |

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

