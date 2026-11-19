#!/bin/bash
# 构建 v18 提交包：sbg3 权重 -> 50 测试帧预测 -> 跨类发射 -> 校验
# 用法: bash scripts/build_v18.sh [ckpt] [outzip]
set -e
cd "$(dirname "$0")/.." || exit 1
PY=D:/app/anaconda/envs/lidar310/python.exe
CKPT=${1:-data/train/ckpt/pp_sbg3_best.pt}
OUT=${2:-data/out/submission_v18.zip}
SYNTH=${SYNTH:-data/synthetic_v13}

# 架构开关必须与训练时一致，否则 load_state_dict 直接报 shape mismatch：
#   deep neck -> backbone.up/up2/up3；WIDE -> b1/b2/b3 双卷积；
#   MSA -> Bus/Truck/Van 各补一个尺寸 anchor，NA 由 14 变 20
export PP_NECK=${PP_NECK:-deep}
export PP_WIDE=${PP_WIDE:-1}
export PP_MSA=${PP_MSA:-1}

echo "[1/3] 预测 50 个官方测试帧 -> _v18raw.zip"
$PY -u scripts/predict_pp.py --ckpt "$CKPT" --out data/out/_v18raw.zip \
    --score-thr 0.02 --topk 3000 --synth-dir "$SYNTH" | tail -3

echo "[2/3] 跨类发射（LH 后验, thr=0.1, scale=0.5, src=0.1）"
$PY -u scripts/launch_json.py --in data/out/_v18raw.zip --out "$OUT" \
    --thr 0.1 --scale 0.5 --src-thr 0.1 | tail -3

echo "[3/3] 校验"
$PY -u scripts/check_submission.py "$OUT" | tail -8
