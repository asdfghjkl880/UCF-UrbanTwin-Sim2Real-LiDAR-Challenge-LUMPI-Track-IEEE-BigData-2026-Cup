#!/usr/bin/env python3
"""
在指定评估基准上跑一个 checkpoint，并调用官方评分程序出分。

用法：
    python scripts/eval_ckpt.py --ckpt data/train/ckpt/pp_v6.pt --bench data/eval_m4_test
    python scripts/eval_ckpt.py --ckpt ... --bench data/reference_data_local --score-thr 0.15

流程：对 bench/detection/*.bin 逐帧推理 -> 组装临时提交 zip（synthetic 用
data/synthetic 的 50 帧）-> 调 lumpi_starting_kit/local_eval.py 出分。
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402
import pp_model as M  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from predict_pp import DECL  # noqa: E402  本地评测也用同一份声明，避免两份不一致


def read_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--bench', default=os.path.join(ROOT, 'data', 'eval_m4_test'))
    ap.add_argument('--score-thr', type=float, default=0.10)
    ap.add_argument('--iou-thr', type=float, default=0.25)
    ap.add_argument('--topk', type=int, default=600)
    ap.add_argument('--synth-dir', default=os.path.join(ROOT, 'data', 'synthetic'))
    ap.add_argument('--tag', default='eval')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.bench, 'config.json')))
    ids = cfg['detection_test_frame_ids']
    det_dir = os.path.join(a.bench, 'detection')
    if not os.path.isdir(det_dir):
        det_dir = a.bench

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M.PointPillars().to(dev)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st['model'])
    model.eval()
    print(f'载入 {os.path.basename(a.ckpt)}（epoch {st.get("epoch")}, '
          f'val mAP {st.get("best", -1):.4f}）  基准 {a.bench}  {len(ids)} 帧')

    preds = {}
    nbox = 0
    for i, fid in enumerate(ids):
        p = os.path.join(det_dir, f'{fid}.bin')
        if not os.path.exists(p):
            continue
        pts = read_bin(p)
        b, s, c = M.detect(model, pts, dev, a.score_thr, a.iou_thr, topk=a.topk)
        preds[fid] = [{'box': [round(float(v), 4) for v in bb],
                       'score': round(float(ss), 4),
                       'class': C.CLASS_NAMES[int(cc)]}
                      for bb, ss, cc in zip(b, s, c)]
        nbox += len(b)
    print(f'推理完成 {len(preds)} 帧 {nbox} 框（{nbox/max(len(preds),1):.0f} 框/帧）')

    out_zip = os.path.join(ROOT, 'data', 'out', f'_{a.tag}.zip')
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    src = sorted(f for f in os.listdir(a.synth_dir) if f.endswith('.bin'))
    n_syn = cfg['n_realism_frames']
    if len(src) < n_syn:
        raise SystemExit(f'{a.synth_dir} 只有 {len(src)} 帧，需要 {n_syn}')
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for k in range(n_syn):
            zf.write(os.path.join(a.synth_dir, src[k]), f'synthetic/{k+1:04d}.bin')
        zf.writestr('predictions.json', json.dumps(preds))
        zf.writestr('declaration.txt', DECL)

    env = os.environ.copy()
    env['TMPDIR'] = env['TEMP'] = env['TMP'] = r'D:\app\anaconda\cache\tmp'
    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, 'lumpi_starting_kit', 'local_eval.py'),
                        out_zip, a.bench],
                       capture_output=True, text=True, env=env,
                       cwd=ROOT)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr)


if __name__ == '__main__':
    main()
