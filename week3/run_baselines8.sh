#!/bin/bash
# 논문 8종 베이스라인 중 나머지 6종을 seed 1·2·3으로 순차 학습 (끝난 것은 건너뜀, 재실행 가능)
cd "$(dirname "$0")"
PY=../.venv/Scripts/python
for seed in 1 2 3; do
  for name in cnn1d tri_ccnn e2cnn cnn_lstm_attn cnn_bilstm convnext; do
    suffix=""; [ $seed -ne 1 ] && suffix="_seed$seed"
    if [ -f "models/${name}${suffix}.pt" ]; then echo "skip $name seed $seed"; continue; fi
    $PY -u w3_step3_baselines.py $name $seed || exit 1
  done
done
echo ALL_TRAINING_DONE
