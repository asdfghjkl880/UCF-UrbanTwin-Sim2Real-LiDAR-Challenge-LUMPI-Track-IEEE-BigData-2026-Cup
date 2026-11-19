#!/bin/bash
# 一键出提交包：官方 50 个测试帧预测 -> 框精修 -> 免标签重排 -> 换声明 -> 校验
# 用法: bash scripts/build_sub.sh <ckpt> <输出zip>
set -e
cd "$(dirname "$0")/.." || exit 1
PY=D:/app/anaconda/envs/lidar310/python.exe
CKPT=${1:-data/train/ckpt/pp_s7_ep07.pt}
OUT=${2:-data/out/submission_v11.zip}
SYNTH=${3:-data/synthetic_v8}
TOPK=${TOPK:-3000}
# 权重是 deep+wide+MSA 训练出来的，架构开关必须与训练一致，否则权重加载会静默失败
export PP_NECK=${PP_NECK:-deep}
export PP_WIDE=${PP_WIDE:-1}
export PP_MSA=${PP_MSA:-1}
export PP_PEAK=${PP_PEAK:-3}

echo "[1/4] 预测 50 个官方测试帧"
$PY -u scripts/predict_pp.py --ckpt "$CKPT" --out data/out/_test_raw.zip \
    --score-thr 0.02 --topk "$TOPK" --flip "${FLIP:-1}" --synth-dir "$SYNTH" | tail -2

echo "[2/4] 几何精修（多候选 + 点支撑择优，按类分治）"
# 2026-11-11 消融（60 合法真实帧，IoU@0.5 召回）：
#   Person 只差中心(.230->.688) / Bicycle+Moto 只差朝向(.254->.570, .195->.683)
#   Truck 差尺寸+中心 / Van 差中心 / Car 已经很好，动它会掉到 .40
# 所以每类只生成它敏感的自由度的候选，再用"框内点数"择优，点支撑不许变差。
$PY -u scripts/refine_zip.py --in data/out/_test_raw.zip \
    --clouds lumpi_starting_kit/detection_test_frames \
    --out data/out/_test_ref.zip | tail -1

echo "[3/4] 免标签重排（score × n/(n+5)，n=框内点数）"
$PY -u - <<'EOF'
import json, zipfile, sys, os
sys.path.insert(0, 'scripts')
import rerank as R
z = zipfile.ZipFile('data/out/_test_ref.zip')
P = json.loads(z.read('predictions.json'))
decl = z.read('declaration.txt').decode('utf-8')
# 必须先删缓存：npts 依赖上一步精修后的框，复用旧缓存会静默错位
if os.path.exists('data/out/_npts_test.npy'):
    os.remove('data/out/_npts_test.npy')
n = R.compute('lumpi_starting_kit/detection_test_frames', P,
              'data/out/_npts_test.npy')
cfg = R.load_cfg()          # reports/rerank_cfg.json（每类 (mode,k,b)，交叉验证得出）
Q = R.rerank_ex(P, n, cfg) if cfg else R.rerank(P, n, 5, 1.0)
print('重排配置', cfg)
syn = sorted(x for x in z.namelist() if x.startswith('synthetic/'))
with zipfile.ZipFile('data/out/_test_final.zip', 'w',
                     zipfile.ZIP_DEFLATED) as o:
    for k, nm in enumerate(syn):
        o.writestr('synthetic/%04d.bin' % (k + 1), z.read(nm))
    o.writestr('predictions.json', json.dumps(Q))
    o.writestr('declaration.txt', decl)
print('重排完成', sum(len(v) for v in Q.values()), '框')
EOF

echo "[4/4] 打包 + 校验"
$PY -u scripts/make_final_sub.py "$OUT" | tail -2
$PY -u scripts/check_submission.py "$OUT" | tail -4
