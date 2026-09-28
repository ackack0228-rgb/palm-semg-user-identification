# 2주차 실습 · 단계 19 (추가, 진단): 스케일64 모델의 5x5 혼동행렬
# step15_confusion.py와 동일한 로직, step17_model_scale64.pt 단일 모델 + step16
# 데이터셋으로 확인 — E->B 혼동이 스케일을 키운 뒤에도 남아있는지 검증한다.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import densenet161

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = ['A', 'B', 'C', 'D', 'E']

d = np.load(os.path.join(BASE_DIR, 'step16_dataset_scale64.npz'))
Xte = torch.from_numpy(d['Xte'])
yte = torch.from_numpy(d['yte'])
loader = DataLoader(TensorDataset(Xte, yte), batch_size=32)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = densenet161(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
model = model.to(dev)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'step17_model_scale64.pt'), map_location=dev))
model.eval()

cm = np.zeros((5, 5), dtype=int)
with torch.no_grad():
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        pred = model(xb).argmax(dim=1)
        for t, p in zip(yb.tolist(), pred.tolist()):
            cm[t, p] += 1

lines = ['confusion matrix (rows=true, cols=pred), step17_model_scale64.pt:']
lines.append('      ' + '  '.join(f'{s:>4}' for s in SUBJECTS))
for i, s in enumerate(SUBJECTS):
    lines.append(f'true {s}: ' + '  '.join(f'{cm[i, j]:>4}' for j in range(5)))
lines.append('')
for i, s in enumerate(SUBJECTS):
    total = cm[i].sum(); correct = cm[i, i]
    errs = [(SUBJECTS[j], cm[i, j]) for j in range(5) if j != i and cm[i, j] > 0]
    errs.sort(key=lambda x: -x[1])
    err_str = ', '.join(f'{n}={c}' for n, c in errs) if errs else 'none'
    lines.append(f'  {s}: correct={correct}/{total}  misclassified_as: {err_str}')

print('\n'.join(lines))
with open(os.path.join(BASE_DIR, 'step19_confusion_matrix_scale64.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('saved: step19_confusion_matrix_scale64.txt')
