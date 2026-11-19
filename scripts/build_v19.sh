#!/bin/bash
# 构建 v19：单次推理 + 镜像 TTA 按类混合（Person/Car 用 TTA） + 跨类发射
# 用法: bash scripts/build_v19.sh [ckpt] [outzip]
set -e
cd "$(dirname "$0")/.." || exit 1
PY=D:/app/anaconda/envs/lidar310/python.exe
CKPT=${1:-data/train/ckpt/pp_sbg3_best.pt}
OUT=${2:-data/out/submission_v19.zip}
SYNTH=${SYNTH:-data/synthetic_v13}

# 架构开关必须与训练一致，否则 load_state_dict 报 shape mismatch
export PP_NECK=${PP_NECK:-deep}
export PP_WIDE=${PP_WIDE:-1}
export PP_MSA=${PP_MSA:-1}

echo "[1/3] 单次推理 -> _v19plain.zip"
$PY -u scripts/predict_pp.py --ckpt "$CKPT" --out data/out/_v19plain.zip \
    --score-thr 0.02 --topk 3000 --synth-dir "$SYNTH" | tail -2

echo "[2/3] 镜像 TTA -> _v19tta.zip"
$PY -u scripts/predict_pp.py --ckpt "$CKPT" --out data/out/_v19tta.zip \
    --score-thr 0.02 --topk 3000 --flip 1 --synth-dir "$SYNTH" | tail -2

echo "[3/3] 按类混合 + 跨类发射 -> $OUT"
$PY -u scripts/mix_json.py --plain data/out/_v19plain.zip --tta data/out/_v19tta.zip \
    --out "$OUT" --tta-classes Person,Car --thr 0.1 --scale 0.5 --src-thr 0.1 | tail -3
$PY -u scripts/check_submission.py "$OUT" | tail -4
