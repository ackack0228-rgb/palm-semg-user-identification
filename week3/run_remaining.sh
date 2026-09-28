#!/usr/bin/env bash
# 3주차 보완 실험 일괄 실행 (강의 PDF 대조 후 추가: 3회 실행 평균 + 10-fold Wilcoxon)
# 순서: 베이스라인 seed 2·3 학습(약 10분) → 3회 평균 평가 → 10-fold + Wilcoxon(약 5시간)
# 각 단계는 이미 끝난 결과가 있으면 건너뛰므로, 중간에 끊겨도 다시 실행하면 이어서 진행된다.
#
# 사용법 (Git Bash, week3 폴더에서): bash run_remaining.sh > run_remaining.log 2>&1
# 진행 확인: tail run_remaining.log  /  grep "accuracy" run_remaining.log
set -e
cd "$(dirname "$0")"
PY=../.venv/Scripts/python

for name in cnn2d resnet18; do
  for seed in 2 3; do
    if [ -f "models/${name}_seed${seed}.pt" ]; then
      echo "[skip] ${name} seed ${seed} 이미 있음"
    else
      $PY -u w3_step3_baselines.py "$name" "$seed"
    fi
  done
done

$PY -u w3_step5_repeat3.py
$PY -u w3_step6_cv10_wilcoxon.py
echo "ALL DONE $(date)"
