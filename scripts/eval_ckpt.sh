#!/bin/bash
# 一键评估某个权重在 holdout 上的完整管线分数（单次 + TTA 混合 + 跨类发射）
# 用法: bash scripts/eval_ckpt.sh <ckpt> <tag>
set -e
cd "$(dirname "$0")/.." || exit 1
PY=D:/app/anaconda/envs/lidar310/python.exe
CKPT=${1:?需要 ckpt 路径}
TAG=${2:-eval}

export PP_NECK=${PP_NECK:-deep}
export PP_WIDE=${PP_WIDE:-1}
export PP_MSA=${PP_MSA:-1}
export OMP_NUM_THREADS=4

echo "[1/3] holdout 单次推理 -> _${TAG}_h.npz"
$PY -u scripts/pred_ada.py --ckpt "$CKPT" --dir data/eval_m4_holdout \
    --out "data/out/_${TAG}_h.npz" --topk 3000 --thr 0.02 2>&1 | tail -2

echo "[2/3] holdout 镜像 TTA -> _${TAG}_tta.npz"
$PY -u scripts/pred_ada.py --ckpt "$CKPT" --dir data/eval_m4_holdout \
    --out "data/out/_${TAG}_tta.npz" --flip 1 --topk 3000 --thr 0.02 2>&1 | tail -2

echo "[3/3] 融合评估"
$PY -u scripts/fuse_final.py --plain "data/out/_${TAG}_h.npz.npy" \
    --tta "data/out/_${TAG}_tta.npz.npy" \
    --out "data/out/_${TAG}_final.npz" --scale 0.5 2>&1 | tail -12
