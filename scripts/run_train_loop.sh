#!/bin/bash
# 训练守护：CUDA 偶发 "illegal instruction" 会打断训练，pp_<tag>.pt 每个 epoch
# 都存了 optimizer/scheduler，所以崩了直接 --resume 续训，代价只有当前 epoch。
TAG=${1:-s9}
EPOCHS=${2:-20}
DATA=${3:-data/train/synth9_tr}
LOG=reports/logs/train_${TAG}.log
cd "$(dirname "$0")/.." || exit 1
for i in $(seq 1 40); do
  D:/app/anaconda/envs/lidar310/python.exe -u scripts/train_pp.py \
    --train-dirs "$DATA" --val-dir data/eval_m4_holdout \
    --epochs "$EPOCHS" --val-every 2 --tag "$TAG" --workers 4 --resume \
    --val-limit "${VAL_LIMIT:-0}" \
    >> "$LOG" 2>&1
  code=$?
  if [ $code -eq 0 ]; then echo "[loop] 正常结束" >> "$LOG"; break; fi
  echo "[loop] 第 $i 次异常退出 code=$code，续训" >> "$LOG"
  sleep 5
done
