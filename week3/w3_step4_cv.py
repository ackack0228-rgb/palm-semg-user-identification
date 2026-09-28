# 3주차 실습 · 단계 4 : 5-fold 교차검증 (DenseNet161)
#
# 핵심: StratifiedKFold에 넣는 것은 "윈도우"가 아니라 "시행(파일)"이다.
#   250개 시행 파일을 먼저 5조각으로 나누고, 각 폴드에서 학습 파일/검증 파일로 따로
#   윈도우+CWT를 만든다 → 같은 시행의 윈도우가 학습과 검증에 동시에 들어갈 수 없다.
#   (2주차 step6의 누수 방지 원칙을 폴드마다 그대로 적용)
# 폴드마다 모델을 새로 만든다(fresh model). 학습 조건은 common.train_model()로 통일.
#
# 장시간(폴드당 약 23분 × 5) 실행이라 폴드가 하나 끝날 때마다 results/cv_folds.json에
# 결과를 저장하고, 다시 실행하면 이미 끝난 폴드는 건너뛴다(중간에 끊겨도 이어서 가능).
#
# 사용법: ../.venv/Scripts/python -u w3_step4_cv.py
# 결과: results/cv_folds.json (폴드별 accuracy / macro F1 / 학습시간, 평균 ± 표준편차)

import os
import glob
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from common import BASE_DIR, WEEK2_DIR, set_seed, build_model, train_model, predict_proba
from step6_dataset import build, subject_of, LABEL_OF_SUBJECT

MODEL = 'densenet161'
SEED = 1
OUT = os.path.join(BASE_DIR, 'results', 'cv_folds.json')


def summarize(folds):
    accs = [f['accuracy'] for f in folds]
    f1s = [f['macro_f1'] for f in folds]
    return {'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s))}


if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    files = sorted(glob.glob(os.path.join(WEEK2_DIR, '..', 'data', 'data', '*', '*.csv')))
    labels = [LABEL_OF_SUBJECT[subject_of(f)] for f in files]
    print('시행 파일 수:', len(files), flush=True)

    state = {'model': MODEL, 'seed': SEED, 'folds': []}
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as f:
            state = json.load(f)
    done = {f['fold'] for f in state['folds']}

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for k, (tr_i, va_i) in enumerate(skf.split(files, labels), start=1):
        if k in done:
            print(f'fold {k}: 이미 완료 — 건너뜀', flush=True)
            continue
        tr_files = [files[i] for i in tr_i]
        va_files = [files[i] for i in va_i]
        assert not set(tr_files) & set(va_files)  # 시행 중복 0건 확인
        Xtr, ytr = build(tr_files)
        Xva, yva = build(va_files)
        print(f'fold {k}: train {Xtr.shape}  val {Xva.shape}', flush=True)

        set_seed(SEED)
        model = build_model(MODEL)  # 폴드마다 새 모델
        model, sec, _ = train_model(model, Xtr, ytr, seed=SEED, tag=f'[fold {k}]')
        pred = predict_proba(model, Xva).argmax(1)
        acc = accuracy_score(yva, pred)
        f1 = f1_score(yva, pred, average='macro')
        print(f'fold {k}: accuracy {acc:.4f}  macro F1 {f1:.4f}  train {sec:.1f}s', flush=True)

        state['folds'].append({'fold': k, 'accuracy': float(acc), 'macro_f1': float(f1),
                               'train_seconds': round(sec, 1), 'n_val': int(len(yva))})
        state['folds'].sort(key=lambda f: f['fold'])
        state['summary'] = summarize(state['folds'])
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        del model, Xtr, Xva

    s = state['summary']
    print(f"5-fold accuracy {s['accuracy_mean'] * 100:.2f} ± {s['accuracy_std'] * 100:.2f} %", flush=True)
    print(f"5-fold macro F1 {s['macro_f1_mean'] * 100:.2f} ± {s['macro_f1_std'] * 100:.2f} %", flush=True)
