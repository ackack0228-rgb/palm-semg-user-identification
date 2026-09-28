# 3주차 실습 · 단계 1 + 2 + 5 : 테스트 평가 · 혼동행렬 · 계산 비용
#
# 같은 테스트셋(2주차 step6_dataset.npz, 50시행 = 950윈도우)에서 각 모델의
#   - Accuracy, macro Precision / Recall / F1, classification_report
#   - 혼동행렬 그림(confusion_<model>.png) + 최대 오류 쌍(대각선 제외 최댓값)
#   - 파라미터 수, 학습 시간, 샘플 1개당 추론 시간(예열 10회 후 100회 평균, CUDA synchronize)
# 을 한 번에 구해 results/metrics.json 과 results/metrics_table.md 로 저장한다.
#
# 평가 대상
#   - cnn2d, resnet18        : w3_step3_baselines.py로 학습 (models/*.pt)
#   - densenet161            : 2주차 step13_model_seed1.pt (같은 레시피, seed=1)
#   - ensemble6 (참고)       : 2주차 최종 6모델 앙상블(step20과 동일 구성)
#
# 사용법: ../.venv/Scripts/python w3_step1_evaluate.py

import os
import re
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.models import densenet161
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from common import BASE_DIR, WEEK2_DIR, SUBJECTS, DEV, build_model, count_params, predict_proba

RES_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'figures')


def load_state(model, path):
    model.load_state_dict(torch.load(path, map_location=DEV))
    return model.to(DEV).eval()


def week2_densenet(ckpt):
    """2주차 체크포인트 로더 (step13/step17은 Dropout 분류기, step10/11은 Linear 단독)."""
    m = densenet161(weights=None)
    if 'step13' in ckpt or 'step17' in ckpt:
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(2208, 5))
    else:
        m.classifier = nn.Linear(2208, 5)
    return load_state(m, os.path.join(WEEK2_DIR, ckpt))


def train_seconds_from_week2_log(logname):
    with open(os.path.join(WEEK2_DIR, logname), encoding='utf-8') as f:
        last = f.read().strip().splitlines()[-1]
    return float(re.search(r'elapsed=([\d.]+)s', last).group(1))


@torch.no_grad()
def inference_ms(models, xs):
    """샘플 1개(batch=1) 예측 시간(ms). models/xs는 앙상블이면 여러 개."""
    for m, x in zip(models, xs):
        m.eval()
    one = [torch.as_tensor(x[:1]).to(DEV) for x in xs]

    def run():
        for m, x in zip(models, one):
            m(x)
    for _ in range(10):  # 예열(warm-up): 첫 호출 오버헤드 제외
        run()
    if DEV == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        run()
    if DEV == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) * 10  # 100회 → 1회당 ms


def max_error_pair(cm):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    return int(i), int(j), int(off[i, j])


def plot_cm(cm, title, path):
    i_err, j_err, _ = max_error_pair(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5), SUBJECTS)
    ax.set_yticks(range(5), SUBJECTS)
    thresh = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else '#1f2937')
    # 최대 오류 쌍 표시 (진한 테두리)
    ax.add_patch(Rectangle((j_err - 0.5, i_err - 0.5), 1, 1, fill=False,
                           edgecolor='#b91c1c', linewidth=2.5))
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(f'{title}\nmax error: {SUBJECTS[i_err]} -> {SUBJECTS[j_err]} ({cm[i_err, j_err]})',
                 fontsize=10)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def metrics(y, pred):
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(5))
    i, j, n = max_error_pair(cm)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision_macro': float(p), 'recall_macro': float(r), 'f1_macro': float(f1),
        'per_class_recall': {s: float(cm[k, k] / cm[k].sum()) for k, s in enumerate(SUBJECTS)},
        'confusion_matrix': cm.tolist(),
        'max_error_pair': {'true': SUBJECTS[i], 'pred': SUBJECTS[j], 'count': n},
        'report': classification_report(y, pred, target_names=SUBJECTS, digits=3),
    }


if __name__ == '__main__':
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    d32 = np.load(os.path.join(WEEK2_DIR, 'step6_dataset.npz'))
    Xte, yte = d32['Xte'], d32['yte']
    print('test:', Xte.shape, ' device:', DEV)

    results = {}

    # --- 단일 모델 3종 (같은 조건) ---
    single = {
        'cnn2d': ('2D CNN', lambda: load_state(build_model('cnn2d'), os.path.join(BASE_DIR, 'models', 'cnn2d.pt')),
                  lambda: json.load(open(os.path.join(RES_DIR, 'train_cnn2d.json')))['train_seconds']),
        'resnet18': ('ResNet18', lambda: load_state(build_model('resnet18'), os.path.join(BASE_DIR, 'models', 'resnet18.pt')),
                     lambda: json.load(open(os.path.join(RES_DIR, 'train_resnet18.json')))['train_seconds']),
        'densenet161': ('DenseNet161', lambda: week2_densenet('step13_model_seed1.pt'),
                        lambda: train_seconds_from_week2_log('step13_train_log_seed1.txt')),
    }
    for key, (label, loader, tsec) in single.items():
        model = loader()
        prob = predict_proba(model, Xte)
        m = metrics(yte, prob.argmax(1))
        m.update(label=label, params=count_params(model), train_seconds=tsec(),
                 inference_ms=inference_ms([model], [Xte]))
        results[key] = m
        plot_cm(np.array(m['confusion_matrix']), label, os.path.join(FIG_DIR, f'confusion_{key}.png'))
        if key == 'densenet161':  # 제출물 ② confusion.png = 제안 모델
            plot_cm(np.array(m['confusion_matrix']), label, os.path.join(FIG_DIR, 'confusion.png'))
        print(f"{label:12s} acc {m['accuracy']:.4f}  F1 {m['f1_macro']:.4f}  "
              f"params {m['params']:,}  train {m['train_seconds']}s  infer {m['inference_ms']:.2f}ms")
        del model
        torch.cuda.empty_cache() if DEV == 'cuda' else None

    # --- 참고: 2주차 최종 6모델 앙상블 (스케일32 DenseNet161 5개 + 스케일64 1개) ---
    d64 = np.load(os.path.join(WEEK2_DIR, 'step16_dataset_scale64.npz'))
    Xte64 = d64['Xte']
    assert np.array_equal(yte, d64['yte'])
    ck32 = ['step10_model_45epoch.pt', 'step11_model_45epoch.pt',
            'step13_model_seed1.pt', 'step13_model_seed2.pt', 'step13_model_seed3.pt']
    models, xs = [], []
    prob = np.zeros((len(yte), 5))
    for ck in ck32:
        mdl = week2_densenet(ck)
        prob += predict_proba(mdl, Xte)
        models.append(mdl); xs.append(Xte)
    mdl = week2_densenet('step17_model_scale64.pt')
    prob += predict_proba(mdl, Xte64)
    models.append(mdl); xs.append(Xte64)
    m = metrics(yte, prob.argmax(1))
    m.update(label='DenseNet161 x6 ensemble', params=sum(count_params(x) for x in models),
             train_seconds=None, inference_ms=inference_ms(models, xs))
    results['ensemble6'] = m
    plot_cm(np.array(m['confusion_matrix']), 'DenseNet161 x6 ensemble', os.path.join(FIG_DIR, 'confusion_ensemble6.png'))
    print(f"ensemble6    acc {m['accuracy']:.4f}  F1 {m['f1_macro']:.4f}  infer {m['inference_ms']:.2f}ms")

    with open(os.path.join(RES_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # README에 붙일 표
    rows = ['| Model | Accuracy | Precision | Recall | F1-score | Params | Train time | Inference / sample |',
            '|---|---|---|---|---|---|---|---|']
    for key, m in results.items():
        t = f"{m['train_seconds']:.0f} s" if m['train_seconds'] else '-'
        rows.append(f"| {m['label']} | {m['accuracy'] * 100:.2f}% | {m['precision_macro'] * 100:.2f}% | "
                    f"{m['recall_macro'] * 100:.2f}% | {m['f1_macro'] * 100:.2f}% | {m['params']:,} | "
                    f"{t} | {m['inference_ms']:.2f} ms |")
    with open(os.path.join(RES_DIR, 'metrics_table.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n\n')
        for key, m in results.items():
            f.write(f"### {m['label']}\n\nmax error pair: {m['max_error_pair']}\n\n```\n{m['report']}\n```\n\n")
    print('saved: results/metrics.json, results/metrics_table.md, figures/confusion_*.png')
