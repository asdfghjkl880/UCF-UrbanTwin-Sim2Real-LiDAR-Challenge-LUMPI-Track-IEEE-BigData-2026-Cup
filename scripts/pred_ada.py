#!/usr/bin/env python3
"""GPU 推理 + AdaBN 域自适应 + 镜像 TTA。

规则允许"用真实训练集做域自适应"（rules.md: Using the train split of the
public real dataset for hyperparameter selection or domain adaptation）。
AdaBN 只重估 BatchNorm 的 running mean/var，**不接触任何标注**，
属于最无争议的域自适应形式。

用法:
    py scripts/pred_ada.py --dir data/eval_m4_holdout \
        --ada data/eval_m4_test/detection --ada-n 24 \
        --out data/out/_rank_ada.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402


def load_pts(p):
    a = np.fromfile(p, dtype=np.float32)
    if a.size % 4 == 0 and a.size % 3 != 0:
        a = a.reshape(-1, 4)[:, :3]
    else:
        a = a.reshape(-1, 3)
    return a


def load_gt(p):
    j = json.load(open(p))
    rows = []
    for o in j.get('objects', j if isinstance(j, list) else []):
        b = o.get('box') or [o.get('x'), o.get('y'), o.get('z'), o.get('l'),
                             o.get('w'), o.get('h'), o.get('heading')]
        nm = o.get('class') or o.get('type') or 'Car'
        if nm not in C.CLASS_NAMES:
            continue
        rows.append([C.CLASS_NAMES.index(nm)] + [float(v) for v in b[:7]])
    return np.array(rows, np.float64).reshape(-1, 8) if rows else np.zeros((0, 8))


def adabn(model, dev, files, momentum=0.05):
    import torch
    import pp_data as D
    n_bn = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = momentum
            n_bn += 1
    model.train()
    used = 0
    with torch.no_grad():
        for p in files:
            pts = load_pts(p)
            feat, coords, npts = D.pillarize(pts)
            if len(feat) == 0:
                continue
            ft = torch.from_numpy(feat)[None].to(dev, non_blocking=True)
            ct = torch.from_numpy(coords).long()[None].to(dev, non_blocking=True)
            nt = torch.from_numpy(npts).long()[None].to(dev, non_blocking=True)
            model(ft, ct, nt)
            used += 1
    model.eval()
    return n_bn, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='data/train/ckpt/pp_s12_best.pt')
    ap.add_argument('--dir', default='data/eval_m4_holdout')
    ap.add_argument('--ada', default='')
    ap.add_argument('--ada-n', type=int, default=24)
    ap.add_argument('--out', default='data/out/_rank_ada.npz')
    ap.add_argument('--topk', type=int, default=3000)
    ap.add_argument('--thr', type=float, default=0.02)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--flip', type=int, default=0, help='1=y 轴镜像 TTA')
    ap.add_argument('--iou-thr', type=float, default=0.25, help='BEV NMS IoU 阈值')
    a = ap.parse_args()

    import torch
    import pp_model as M
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M.PointPillars().to(dev).eval()
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st['model'])
    print(f'载入 {a.ckpt}  device={dev}', flush=True)

    if a.ada:
        files = sorted(glob.glob(os.path.join(a.ada, '*.bin')))
        rng = np.random.default_rng(0)
        if len(files) > a.ada_n:
            files = [files[i] for i in sorted(rng.choice(len(files), a.ada_n,
                                                         replace=False))]
        nb, used = adabn(model, dev, files)
        print(f'AdaBN: {nb} 个 BN 层, {used} 帧真实点云（无标注）', flush=True)

    from diag_rank import _frames_from_dir
    frames = list(_frames_from_dir(a.dir, a.limit))
    out = {}
    for i, (fid, pts, gt) in enumerate(frames):
        if a.flip:
            b, s, c = M.detect_tta(model, pts, dev, a.thr, a.iou_thr, topk=a.topk,
                                   flips=(1,))
        else:
            b, s, c = M.detect(model, pts, dev, a.thr, a.iou_thr, topk=a.topk)
        out[fid] = (b.astype(np.float32), s.astype(np.float32),
                    c.astype(np.int64), gt)
        if i % 10 == 0:
            print(f'  [{i+1}/{len(frames)}] {fid} {len(b)} 框', flush=True)
    np.save(a.out, out, allow_pickle=True)
    print('写出', a.out)


if __name__ == '__main__':
    main()
