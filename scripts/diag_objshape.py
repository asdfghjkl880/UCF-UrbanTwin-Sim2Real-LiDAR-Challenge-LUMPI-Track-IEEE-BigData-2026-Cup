#!/usr/bin/env python3
"""诊断：真实目标与合成目标的点模式差异。

只做"观察"，不参与训练。真实侧用非禁用帧（test_data 的 000000~000030，
不在 forbidden_frames.txt 里）的标注框，纯统计，不喂给模型。

对每个框算一组签名：
  n          框内点数（z > -1.6，排除切进框里的地面点）
  top        lz > 0.45H 的比例（顶面）
  sx / sy    点在 |lx| > 0.45L / |ly| > 0.45W 的比例（两侧面）
  front      朝向传感器那一半的点比例（真实应显著 > 0.5，背阴面看不见）
  nn         框内最近邻间距中位数（点阵疏密）
  zfill      竖直方向被覆盖的高度 / H

用法：
  python scripts/diag_objshape.py --real data/eval_m4_holdout --n 30
  python scripts/diag_objshape.py --shard data/train/synth6_va --n 30
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z_AIR = -1.60


def local(pts, b):
    cx, cy, cz, L, W, H, hd = b
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    c, s = np.cos(-hd), np.sin(-hd)
    return (dx * c - dy * s, dx * s + dy * c, pts[:, 2] - cz)


def in_box(pts, b):
    lx, ly, lz = local(pts, b)
    return ((np.abs(lx) <= b[3] / 2) & (np.abs(ly) <= b[4] / 2) &
            (np.abs(lz) <= b[5] / 2) & (pts[:, 2] > Z_AIR))


def nn_dist(p):
    """框内最近邻间距中位数（子采样到 200 点控制开销）。"""
    if len(p) < 3:
        return np.nan
    if len(p) > 200:
        p = p[np.random.choice(len(p), 200, replace=False)]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def signature(pts, b):
    m = in_box(pts, b)
    if m.sum() < 3:
        return None
    p = pts[m]
    lx, ly, lz = local(p, b)
    L, W, H = b[3], b[4], b[5]
    # 传感器在原点；对象中心的方位角方向
    az = np.arctan2(b[1], b[0])
    rx, ry = np.cos(az), np.sin(az)
    facing = ((p[:, 0] * rx + p[:, 1] * ry) >
              (b[0] * rx + b[1] * ry))
    return dict(
        n=int(m.sum()),
        top=float(np.mean(lz > 0.45 * H)),
        sx=float(np.mean(np.abs(lx) > 0.45 * L)),
        sy=float(np.mean(np.abs(ly) > 0.45 * W)),
        front=float(np.mean(facing)),
        nn=nn_dist(p),
        zfill=float((lz.max() - lz.min()) / max(H, 1e-6)),
        xfill=float(2.0 * np.percentile(np.abs(lx), 98) / max(L, 1e-6)),
        yfill=float(2.0 * np.percentile(np.abs(ly), 98) / max(W, 1e-6)),
        ofs=float(np.mean(np.abs(lx)) / max(L, 1e-6)),
    )


def load_real(d, n):
    gt = json.load(open(os.path.join(d, 'detection_gt.json'), encoding='utf-8'))
    out = []
    for fid in sorted(gt)[:n]:
        p = os.path.join(d, f'{fid}.bin')
        if not os.path.exists(p):
            continue
        out.append((np.fromfile(p, np.float32).reshape(-1, 3), gt[fid]))
    return out


def load_shard(d, n):
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('_meta.json'):
            continue
        meta = json.load(open(os.path.join(d, fn), encoding='utf-8'))
        arr = np.load(os.path.join(d, fn.replace('_meta.json', '.npy')))
        off = meta['offsets']
        for i in range(len(off) - 1):
            if len(out) >= n:
                return out
            out.append((arr[off[i]:off[i + 1]],
                        [{'box': [float(v) for v in r[1:8]],
                          'class': C.CLASS_NAMES[int(r[0])]}
                         for r in meta['labels'][i]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default='')
    ap.add_argument('--shard', default='')
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    frames = load_real(a.real, a.n) if a.real else load_shard(a.shard, a.n)
    print(f'来源 {"real:" + a.real if a.real else "synth:" + a.shard}  帧 {len(frames)}')

    agg = {}
    for pts, gts in frames:
        for g in gts:
            s = signature(pts, g['box'])
            if s:
                agg.setdefault(g['class'], []).append(s)

    hdr = (f'{"class":<11}{"n_box":>7}{"n_pts":>7}{"top":>7}{"sx":>7}'
           f'{"sy":>7}{"front":>7}{"nn":>7}{"zfill":>7}{"xfill":>7}'
           f'{"yfill":>7}')
    print(hdr)
    print('-' * len(hdr))
    keys = ('n', 'top', 'sx', 'sy', 'front', 'nn', 'zfill', 'xfill', 'yfill')
    for cn in C.CLASS_NAMES:
        sub = agg.get(cn)
        if not sub:
            continue
        med = [float(np.median([s[k] for s in sub])) for k in keys]
        print(f'{cn:<11}{len(sub):>7}' + ''.join(f'{v:>7.3f}' for v in med))
    allsub = [s for v in agg.values() for s in v]
    med = [float(np.median([s[k] for s in allsub])) for k in keys]
    print(f'{"ALL":<11}{len(allsub):>7}' + ''.join(f'{v:>7.3f}' for v in med))


if __name__ == '__main__':
    main()
