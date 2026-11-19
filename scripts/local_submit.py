#!/usr/bin/env python3
"""
生成本地评分包并用官方评分器打分。

为什么需要单独一个脚本：本地基准只有 31 帧 realism 参考（`reference_data_local`），
而 score.py 要求 `synthetic/` 的帧数 == `config["n_realism_frames"]`，
所以本地包必须是 31 帧、且 predictions 的键必须是本地那 31 个帧 ID。
正式提交则是 50 帧 / 50 个测试帧 ID（见 submission_format.md）。

    python scripts/local_submit.py --ckpt data/train/ckpt/pp_v5.pt
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LOCAL_BENCH, ROOT  # noqa: E402
import pp_config as C  # noqa: E402
import pp_model as M  # noqa: E402
from scene_gen import gen_frame  # noqa: E402
from train_pp import to_eval_format  # noqa: E402

DECL = ("I confirm that I did not train on any frame listed in forbidden_frames.txt.\n"
        "Training data: 4000 synthetic frames from my own scene generator + the real\n"
        "frames shipped in test_data (000000-000030, not in forbidden_frames.txt).\n"
        "Team: local-dev\nDate: 2026-11-09\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--score-thr', type=float, default=0.10)
    ap.add_argument('--iou-thr', type=float, default=0.25)
    ap.add_argument('--topk', type=int, default=600, help='每类保留的候选数')
    ap.add_argument('--moto', type=float, default=0.5)
    ap.add_argument('--no-snap', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'local_submission.zip'))
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(LOCAL_BENCH, 'config.json'), encoding='utf-8'))
    ids = [str(i) for i in cfg['detection_test_frame_ids']]
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M.PointPillars().to(dev)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st['model'])
    model.eval()
    print(f'载入 {a.ckpt}（epoch {st.get("epoch")}, val mAP {st.get("best", -1):.4f}）')

    # 31 帧合成点云（用本地帧 ID 命名，和 predictions 的键对齐）。
    # 必须显式 align='affine'：这批帧只用来评 realism，不参与任何推理，
    # 而 affine 把各轴 (mu, sd) 都对齐到真实帧，实测 realism 0.8368 -> 0.8772
    # （FPD 8.52 -> 2.18）。检测那侧走的是 val_real 真实帧，不受影响。
    syn = {}
    for i, fid in enumerate(ids):
        pts, _ = gen_frame(seed=900000 + i, align='affine')
        syn[fid] = pts.astype(np.float32)

    preds = {}
    for fid in ids:
        # 推理必须用**真实验证帧**，不能用上面为 realism 生成的那批合成帧 ——
        # 拿合成帧的预测去对真实 GT 打分，AP 恒为 0（这个 bug 藏了很久，
        # 表现为 score_detection=0 而 realism 正常）。
        p = os.path.join(C.VAL_REAL, fid + '.bin')
        if os.path.isfile(p):
            pts = np.fromfile(p, dtype=np.float32).reshape(-1, 3)
        else:
            pts = syn[fid]
        b, s, c = M.detect(model, pts, dev, a.score_thr, a.iou_thr,
                           topk=a.topk, moto_from_bike=a.moto,
                           snap_z=not a.no_snap)
        preds[fid] = to_eval_format(b, s, c)

    with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fid in ids:
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
                syn[fid].tofile(f.name)
                tmp = f.name
            zf.write(tmp, f'synthetic/{fid}.bin')
            os.remove(tmp)
        zf.writestr('predictions.json', json.dumps(preds))
        zf.writestr('declaration.txt', DECL)
    print(f'打包 {a.out} ({os.path.getsize(a.out)/2**20:.1f} MB)')

    print('\n--- 官方评分器 ---')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'lumpi_starting_kit', 'local_eval.py'),
                        a.out, LOCAL_BENCH], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    for line in (r.stdout or '').splitlines():
        if any(k in line for k in ('score_', 'mAP', 'CD', 'MMD', 'EMD', 'FPD',
                                   'AP_3d_0.5_R40_')):
            print(line.strip())


if __name__ == '__main__':
    main()
