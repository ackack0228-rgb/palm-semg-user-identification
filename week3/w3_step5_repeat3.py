# 3주차 보완 · 3회 실행 평균 (강의 PDF 6쪽 순위 산정 규칙: "동일 분할 · 3회 실행 평균")
#
# 같은 테스트셋(step6_dataset.npz, 950윈도우)에서 모델별로 seed 1, 2, 3 세 번 학습한
# 체크포인트를 평가해 Accuracy / macro F1의 평균 ± 표준편차를 구한다.
#   - 베이스라인 8종  : models/<name>.pt (seed 1), models/<name>_seed{2,3}.pt
#                       (w3_step3_baselines.py <name> 2 / 3 으로 먼저 학습)
#   - densenet161     : 2주차 step13_model_seed{1,2,3}.pt (같은 레시피)
# 표준편차는 5-fold 표와 같은 방식(모표준편차, np.std)으로 계산한다.
#
# 사용법: ../.venv/Scripts/python w3_step5_repeat3.py
# 결과: results/repeat3.json, results/repeat3_table.md

import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from common import BASE_DIR, WEEK2_DIR, DEV, RAW_INPUT_MODELS, build_model, predict_proba, load_split
from w3_step1_evaluate import load_state, week2_densenet, BASELINES

SEEDS = [1, 2, 3]
LABELS = {**BASELINES, 'densenet161': 'DenseNet161'}


def load(name, seed):
    if name == 'densenet161':
        return week2_densenet(f'step13_model_seed{seed}.pt')
    fname = f'{name}.pt' if seed == 1 else f'{name}_seed{seed}.pt'
    return load_state(build_model(name), os.path.join(BASE_DIR, 'models', fname))


if __name__ == '__main__':
    d = np.load(os.path.join(WEEK2_DIR, 'step6_dataset.npz'))
    Xte, yte = d['Xte'], d['yte']
    print('test:', Xte.shape, ' device:', DEV)

    results = {}
    for name, label in LABELS.items():
        runs = []
        for seed in SEEDS:
            X = load_split(name)[2] if name in RAW_INPUT_MODELS else Xte
            pred = predict_proba(load(name, seed), X).argmax(1)
            runs.append({'seed': seed, 'accuracy': float(accuracy_score(yte, pred)),
                         'macro_f1': float(f1_score(yte, pred, average='macro'))})
            print(f"{label:12s} seed {seed}: acc {runs[-1]['accuracy']:.4f}  F1 {runs[-1]['macro_f1']:.4f}")
        accs = [r['accuracy'] for r in runs]
        f1s = [r['macro_f1'] for r in runs]
        results[name] = {'label': label, 'runs': runs,
                         'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
                         'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s))}

    with open(os.path.join(BASE_DIR, 'results', 'repeat3.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    rows = ['| Model | seed 1 | seed 2 | seed 3 | **Accuracy 평균 ± 표준편차** | **macro F1 평균 ± 표준편차** |',
            '|---|---|---|---|---|---|']
    for r in results.values():
        seeds = ' | '.join(f"{x['accuracy'] * 100:.2f}%" for x in r['runs'])
        rows.append(f"| {r['label']} | {seeds} | **{r['accuracy_mean'] * 100:.2f} ± {r['accuracy_std'] * 100:.2f}%** | "
                    f"**{r['macro_f1_mean'] * 100:.2f} ± {r['macro_f1_std'] * 100:.2f}%** |")
    with open(os.path.join(BASE_DIR, 'results', 'repeat3_table.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')
    print('\n'.join(rows))
    print('saved: results/repeat3.json, results/repeat3_table.md')
