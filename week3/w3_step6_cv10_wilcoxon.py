# 3주차 보완 · 10-fold 교차검증 + Wilcoxon 부호순위 검정 (강의 PDF 48쪽, 논문 Table 5)
#
# 왜 10-fold인가: Wilcoxon 부호순위 검정은 fold 수가 n이면 양측 p값의 최솟값이 2 / 2^n이다.
#   n = 5  → 최솟값 0.0625  : 모든 fold에서 이겨도 p < 0.05가 원리적으로 불가능
#   n = 10 → 최솟값 0.00195 : 검정이 의미를 가진다 (논문도 10-fold 결과로 검정)
#
# 방법
#   - 250개 시행 파일을 StratifiedKFold(10, shuffle=True, random_state=42)로 나눈다
#     (w3_step4_cv.py와 같은 원칙: 윈도우가 아니라 "시행"을 나눔 → 누수 없음).
#   - 한 fold 안에서 네 모델(cnn1d, cnn2d, resnet18, densenet161)을 같은 학습/검증 시행으로 학습한다
#     → fold별로 짝지어진(paired) 결과가 나와 Wilcoxon 부호순위 검정을 쓸 수 있다.
#   - 학습 조건은 common.train_model() (45에폭, seed 1)로 세 모델 모두 동일.
#   - 비교 지표: 논문과 같이 정확도와 가중(weighted) F1. macro F1도 함께 저장.
#
# 장시간 실행(총 약 5시간: DenseNet161 fold당 약 26분)이라 (fold, model) 하나가 끝날 때마다
# results/cv10_folds.json에 저장하고, 다시 실행하면 끝난 것은 건너뛴다.
# 모든 fold가 끝나면 Wilcoxon 검정을 수행해 results/wilcoxon.json, results/wilcoxon_table.md 저장.
#
# 사용법: ../.venv/Scripts/python -u w3_step6_cv10_wilcoxon.py > cv10_run.log 2>&1
#         (중간 결과만으로 검정 결과를 다시 보고 싶으면 --report)

import os
import sys
import glob
import json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from common import (BASE_DIR, WEEK2_DIR, RAW_INPUT_MODELS, set_seed, build_model, train_model,
                    predict_proba, build_raw)
from step6_dataset import build, subject_of, LABEL_OF_SUBJECT

K = 10
SEED = 1
# 1D CNN은 나중에 추가(단일 split에서 DenseNet161보다 높게 나와 같은 fold로 짝지어 검정하려고).
# 같은 파일 목록 + 같은 random_state라 fold 구성은 먼저 끝낸 세 모델과 완전히 같다.
MODELS = ['cnn1d', 'cnn2d', 'resnet18', 'densenet161']  # 빠른 모델부터 (fold 하나 안에서)
PROPOSED = 'densenet161'
LABELS = {'cnn1d': '1D CNN', 'cnn2d': '2D CNN', 'resnet18': 'ResNet18', 'densenet161': 'DenseNet161'}
OUT = os.path.join(BASE_DIR, 'results', 'cv10_folds.json')


def save(state):
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def by_model(state, name):
    rows = sorted((r for r in state['runs'] if r['model'] == name), key=lambda r: r['fold'])
    return rows


def report(state):
    res = {'k': K, 'summary': {}, 'tests': []}
    for name in MODELS:
        rows = by_model(state, name)
        if rows:
            acc = [r['accuracy'] for r in rows]
            wf1 = [r['weighted_f1'] for r in rows]
            res['summary'][name] = {'n_folds': len(rows),
                                    'accuracy_mean': float(np.mean(acc)), 'accuracy_std': float(np.std(acc)),
                                    'weighted_f1_mean': float(np.mean(wf1)), 'weighted_f1_std': float(np.std(wf1))}
    prop = {r['fold']: r for r in by_model(state, PROPOSED)}
    for name in MODELS:
        if name == PROPOSED:
            continue
        base = {r['fold']: r for r in by_model(state, name)}
        folds = sorted(set(prop) & set(base))
        if len(folds) < K:
            continue  # 모든 fold가 끝나야 검정
        for metric in ['accuracy', 'weighted_f1']:
            a = np.array([prop[k][metric] for k in folds])
            b = np.array([base[k][metric] for k in folds])
            two = wilcoxon(a, b)                          # 양측 검정 (논문과 같은 기준 p < 0.05)
            one = wilcoxon(a, b, alternative='greater')   # 참고: "제안 모델이 더 크다" 단측
            # (베이스라인이 더 높으면 이 단측 p는 1에 가깝다 — 방향은 표의 '더 높은 쪽' 칸으로 본다)
            res['tests'].append({'baseline': name, 'metric': metric, 'n': len(folds),
                                 'wins': int((a > b).sum()), 'ties': int((a == b).sum()),
                                 'mean_diff': float((a - b).mean()),
                                 'statistic': float(two.statistic), 'p_two_sided': float(two.pvalue),
                                 'p_one_sided': float(one.pvalue)})
    with open(os.path.join(BASE_DIR, 'results', 'wilcoxon.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)

    lines = [f'{K}-fold 교차검증 (시행 단위, 모든 모델 같은 fold)', '',
             '| Model | Accuracy 평균 ± 표준편차 | Weighted F1 평균 ± 표준편차 | 완료 fold |',
             '|---|---|---|---|']
    for name, s in res['summary'].items():
        lines.append(f"| {LABELS[name]} | {s['accuracy_mean'] * 100:.2f} ± {s['accuracy_std'] * 100:.2f}% | "
                     f"{s['weighted_f1_mean'] * 100:.2f} ± {s['weighted_f1_std'] * 100:.2f}% | {s['n_folds']}/{K} |")
    if res['tests']:
        lines += ['', 'Wilcoxon 부호순위 검정 (DenseNet161 vs 베이스라인, 유의수준 0.05)', '',
                  '| 비교 | 지표 | DenseNet161 승 / fold | 평균 차이 | 양측 p | 단측 p | 유의 | 더 높은 쪽 |',
                  '|---|---|---|---|---|---|---|---|']
        for t in res['tests']:
            lines.append(f"| DenseNet161 vs {LABELS[t['baseline']]} | {t['metric']} | {t['wins']} / {t['n']} | "
                         f"{t['mean_diff'] * 100:+.2f}%p | {t['p_two_sided']:.4f} | {t['p_one_sided']:.4f} | "
                         f"{'O' if t['p_two_sided'] < 0.05 else 'X'} | "
                         f"{'DenseNet161' if t['mean_diff'] > 0 else LABELS[t['baseline']]} |")
    with open(os.path.join(BASE_DIR, 'results', 'wilcoxon_table.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    state = {'k': K, 'seed': SEED, 'runs': []}
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as f:
            state = json.load(f)
    if '--report' in sys.argv:
        report(state)
        sys.exit(0)

    files = sorted(glob.glob(os.path.join(WEEK2_DIR, '..', 'data', 'data', '*', '*.csv')))
    labels = [LABEL_OF_SUBJECT[subject_of(f)] for f in files]
    print('시행 파일 수:', len(files), flush=True)

    done = {(r['fold'], r['model']) for r in state['runs']}
    skf = StratifiedKFold(K, shuffle=True, random_state=42)
    for k, (tr_i, va_i) in enumerate(skf.split(files, labels), start=1):
        todo = [m for m in MODELS if (k, m) not in done]
        if not todo:
            print(f'fold {k}: 이미 완료 — 건너뜀', flush=True)
            continue
        tr_files = [files[i] for i in tr_i]
        va_files = [files[i] for i in va_i]
        assert not set(tr_files) & set(va_files)  # 시행 중복 0건 확인
        data = {}
        if any(m in RAW_INPUT_MODELS for m in todo):
            data['raw'] = build_raw(tr_files) + build_raw(va_files)
        if any(m not in RAW_INPUT_MODELS for m in todo):
            data['cwt'] = build(tr_files) + build(va_files)
        print(f'fold {k}: val {len(va_files)} trials  todo {todo}', flush=True)

        for name in todo:
            Xtr, ytr, Xva, yva = data['raw' if name in RAW_INPUT_MODELS else 'cwt']
            set_seed(SEED)
            model = build_model(name)  # fold·모델마다 새 모델
            model, sec, _ = train_model(model, Xtr, ytr, seed=SEED, tag=f'[fold {k} {name}]')
            pred = predict_proba(model, Xva).argmax(1)
            row = {'fold': k, 'model': name,
                   'accuracy': float(accuracy_score(yva, pred)),
                   'weighted_f1': float(f1_score(yva, pred, average='weighted')),
                   'macro_f1': float(f1_score(yva, pred, average='macro')),
                   'train_seconds': round(sec, 1), 'n_val': int(len(yva))}
            print(f"fold {k} {name}: accuracy {row['accuracy']:.4f}  weighted F1 {row['weighted_f1']:.4f}  "
                  f"train {sec:.1f}s", flush=True)
            state['runs'].append(row)
            save(state)
            del model
        del data

    report(state)
