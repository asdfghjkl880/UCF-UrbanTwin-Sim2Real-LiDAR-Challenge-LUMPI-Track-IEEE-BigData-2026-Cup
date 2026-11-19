#!/bin/bash
# 大训练启动器：22k 帧合成数据 + POS_R=2 + 深宽 MSA 架构，从 s13a_ep03 热启动。
# CUDA 偶发 illegal instruction，所以外面套一层 --resume 续训。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=D:/app/anaconda/envs/lidar310/python.exe
export PP_NECK=${PP_NECK:-deep}
export PP_WIDE=${PP_WIDE:-1}
export PP_MSA=${PP_MSA:-1}
export PP_PEAK=${PP_PEAK:-3}
export PP_POS_R=${PP_POS_R:-2}
export TMPDIR=D:/app/anaconda/cache/tmp
export TEMP='D:\app\anaconda\cache\tmp'
export TMP='D:\app\anaconda\cache\tmp'

TAG=${TAG:-s15}
EPOCHS=${EPOCHS:-30}
INIT=${INIT:-data/train/ckpt/pp_s13a_ep03.pt}
DIRS=${DIRS:-"data/train/s11,data/train/s14p0,data/train/s14p1,data/train/s14p2,data/train/s14p3,data/train/s14p4,data/train/s14p5,data/train/s14p6,data/train/s14p7,data/train/s14p8,data/train/s14p9"}
LOG=reports/logs/train_${TAG}.log

EXTRA=""
[ -f "data/train/ckpt/pp_${TAG}.pt" ] && EXTRA="--resume"
if [ -z "$EXTRA" ]; then
  INITARG="--init $INIT"
else
  INITARG=""
fi

for i in $(seq 1 40); do
  $PY -u scripts/train_pp.py \
    --train-dirs "$DIRS" --val-dir data/eval_m4_holdout \
    --epochs "$EPOCHS" --val-every 2 --tag "$TAG" --workers "${WORKERS:-4}" \
    $INITARG $EXTRA >> "$LOG" 2>&1
  code=$?
  if [ $code -eq 0 ]; then echo "[loop] 正常结束" >> "$LOG"; break; fi
  echo "[loop] 第 $i 次异常退出 code=$code，续训" >> "$LOG"
  EXTRA="--resume"; INITARG=""
  sleep 10
done
