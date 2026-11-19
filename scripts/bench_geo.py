#!/usr/bin/env python3
"""几何精修 A/B 台（官方 40 点 AP 口径）。

在缓存的原始预测上跑不同精修变体，输出逐类 recall@0.5 与 AP40。
用法:
    py scripts/bench_geo.py                       # 默认几组
    py scripts/bench_geo.py --variants none,v2,per
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lumpi_starting_kit', 'scoring_program'))

import pp_config as C  # noqa: E402
import detection_eval as DE  # noqa: E402
import geo_fit as G  # noqa: E402

NAMES = C.CLASS_NAMES
R40 = np.linspace(1.0 / 40.0, 1.0, 40)

# 各组：作用于哪些类
SETS = {
    'per': ('Person',),
    'veh': ('Car', 'Van', 'Truck', 'Bus'),
    'two': ('Bicycle', 'Motorcycle'),
    'small': ('Person', 'Bicycle', 'Motorcycle'),
    'all': tuple(NAMES),
}


def ap40(scores, tps, n_gt):
    if len(scores) == 0 or n_gt == 0:
        return 0.0
    o = np.argsort(-scores, kind='stable')
    t = tps[o]
    tp = np.cumsum(t)
    fp = np.cumsum(~t)
    rec = tp / float(n_gt)
    prec = tp / np.maximum(tp + fp, 1)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, R40, side='left')
    p = np.where(idx < len(env), env[np.clip(idx, 0, len(env) - 1)], 0.0)
    p = np.where(rec[-1] + 1e-12 >= R40, p, 0.0)
    return float(p.mean())


def greedy_tp(iou_pairs, npred, ngt, w, thr=0.5):
    a, b, v = iou_pairs
    tps = np.zeros(npred, bool)
    if len(a) == 0:
        return tps
    order = np.argsort(-w, kind='stable')
    adj = {}
    for aa, bb, vv in zip(a, b, v):
        if vv >= thr:
            adj.setdefault(int(aa), []).append((int(bb), float(vv)))
    used = np.zeros(ngt, bool)
    for p in order:
        lst = adj.get(int(p))
        if not lst:
            continue
        best = None
        for g, vv in lst:
            if not used[g] and (best is None or vv > best[1]):
                best = (g, vv)
        if best is not None:
            used[best[0]] = True
            tps[p] = True
    return tps


def iou_pairs(pb, gb):
    """返回 (pred_idx, gt_idx, iou) 三元组，已按中心距门限裁剪。"""
    if len(pb) == 0 or len(gb) == 0:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    d = np.hypot(pb[:, None, 0] - gb[None, :, 0], pb[:, None, 1] - gb[None, :, 1])
    diag = 0.5 * np.hypot(gb[:, 3], gb[:, 4])[None, :] + \
           0.5 * np.hypot(pb[:, 3], pb[:, 4])[:, None]
    a, b = np.nonzero(d < diag * 1.1)
    ious = np.array([DE._iou_3d(gb[b[k]], pb[a[k]]) for k in range(len(a))])
    return a, b, ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/out/_rank_s12.npz')
    ap.add_argument('--clouds', default='data/eval_m4_holdout')
    ap.add_argument('--variants', default='none,v2,per,veh,two,small')
    a = ap.parse_args()

    z = np.load(a.cache, allow_pickle=True)
    if not hasattr(z, 'keys'):
        z = z.item()
    fids = sorted(z.keys())
    names = a.variants.split(',')
    res = {}
    for vn in names:
        if vn == 'none':
            sets = ()
        elif vn == 'v2':
            sets = SETS['all']
        else:
            sets = SETS[vn]
        per, rec = {}, {}
        for nm in NAMES:
            S, T, ng, hit = [], [], 0, 0
            for fid in fids:
                pb, sc, cl, gt = z[fid]
                pb = np.asarray(pb, np.float64)
                cl = np.asarray(cl).astype(int)
                gt = np.asarray(gt, np.float64).reshape(-1, 8)
                gi = np.flatnonzero(gt[:, 0].astype(int) == NAMES.index(nm))
                ci = NAMES.index(nm)
                pi = np.flatnonzero(cl == ci)
                if len(gi) == 0:
                    continue
                ng += len(gi)
                gb = gt[gi][:, 1:8]
                if len(pi) == 0:
                    continue
                P = pb[pi].copy()
                if sets:
                    p = os.path.join(a.clouds, fid + '.bin')
                    pts = np.fromfile(p, dtype=np.float32).reshape(-1, 3).astype(np.float64) \
                        if os.path.exists(p) else np.zeros((0, 3))
                    if len(pts) > 0:
                        P = G.fit_frame(pts, P, np.full(len(P), ci), apply_to=sets,
                                        tree=cKDTree(pts))
                pairs = iou_pairs(P, gb)
                aa, bb, vv = pairs
                if len(vv):
                    mx = np.zeros(len(gb))
                    np.maximum.at(mx, bb, vv)
                    hit += int((mx >= 0.5).sum())
                w = np.asarray(sc, np.float64)[pi]
                S.append(w)
                T.append(greedy_tp(pairs, len(P), len(gb), w))
            if ng == 0:
                per[nm] = 0.0
                rec[nm] = 0.0
                continue
            S = np.concatenate(S) if S else np.zeros(0)
            T = np.concatenate(T) if T else np.zeros(0, bool)
            per[nm] = ap40(S, T, ng)
            rec[nm] = hit / ng
        res[vn] = (per, rec)
        live = [n for n in NAMES if rec[n] > 0 or per[n] > 0]
        m = float(np.mean([per[n] for n in live]))
        print(f'{vn:6s} mAP {m:.4f} | ' +
              ' '.join(f'{n[:3]} {per[n]:.3f}/{rec[n]:.2f}' for n in live), flush=True)

    print(f'\n{"variant":8s} {"mAP":>7s} ' +
          ' '.join(f'{n[:3]:>11s}' for n in NAMES if n != 'Unknown'))
    for vn in names:
        per, rec = res[vn]
        live = [n for n in NAMES if n != 'Unknown']
        m = float(np.mean([per[n] for n in live]))
        print(f'{vn:8s} {m:7.4f} ' +
              ' '.join(f'{per[n]:5.3f}/{rec[n]:.2f}' for n in live))


if __name__ == '__main__':
    main()
