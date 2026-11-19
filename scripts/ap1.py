#!/usr/bin/env python3
"""单类 11 点 AP 的快速评估 + 重打分参数搜索。

官方 `_compute_ap_11` 就是对 recall ∈ {0,0.1,...,1.0} 取 precision 包络，
所以只要拿到每帧"预测按分数排序后逐个贪心匹配 GT"得到的 (score, is_tp)，
全类合并排序就能算 AP。把 IoU 矩阵**预先算一次**，之后换分数权重只重跑贪心，
一次搜索从几十分钟压到几十秒。

用法:
    py scripts/ap1.py --build              # 预计算 IoU 矩阵（每类一次）
    py scripts/ap1.py                      # 搜索重打分参数
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lumpi_starting_kit', 'scoring_program'))

import pp_config as C  # noqa: E402
import detection_eval as DE  # noqa: E402

NAMES = C.CLASS_NAMES
R11 = np.arange(11) / 10.0
R40 = np.linspace(1.0 / 40.0, 1.0, 40)


def ap11(scores, tps, n_gt):
    """scores/tps: 同类全部预测，已同序；n_gt: 该类 GT 总数（召回分母）。"""
    if len(scores) == 0 or n_gt == 0:
        return 0.0
    o = np.argsort(-scores, kind='stable')
    t = tps[o]
    tp = np.cumsum(t)
    fp = np.cumsum(1 - t)
    rec = tp / float(n_gt)
    prec = tp / np.maximum(tp + fp, 1)
    # precision 包络（从后往前取最大）
    env = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, R11, side='left')
    p = np.where(idx < len(env), env[np.clip(idx, 0, len(env) - 1)], 0.0)
    # recall 达不到的档位精度记 0
    p = np.where(rec[-1] + 1e-12 >= R11, p, 0.0)
    return float(p.mean())


def box_feats(pts, box):
    x, y, z, dx, dy, dz, h = box
    c, s = np.cos(h), np.sin(h)
    px = pts[:, 0] - x
    py = pts[:, 1] - y
    lx = c * px + s * py
    ly = -s * px + c * py
    m = (np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2)
    m &= (pts[:, 2] >= z - dz / 2) & (pts[:, 2] <= z + dz / 2)
    n = int(m.sum())
    if n < 3:
        return (n, 0.0, 0.0, 0.0)
    xy = np.stack([lx[m], ly[m]], 1) - np.stack([lx[m], ly[m]], 1).mean(0)
    cov = xy.T @ xy / max(n - 1, 1)
    w = np.clip(np.linalg.eigvalsh(cov), 1e-9, None)
    zz = pts[m, 2]
    return (n, float(n / max(dx * dy, 1e-6)),
            float(np.sqrt(w[0] / w[1])),
            float((zz.max() - zz.min()) / max(dz, 1e-6)))


def build(cache, clouds, out, max_iou_pairs=4):
    z = np.load(cache, allow_pickle=True)
    data = {}
    for fid in z.keys():
        pb, sc, cl, gt = z[fid]
        pb = np.asarray(pb, np.float64)
        G = [{'box': [float(v) for v in r[1:8]], 'class': NAMES[int(r[0])]}
             for r in gt]
        p = os.path.join(clouds, fid + '.bin')
        pts = (np.fromfile(p, dtype=np.float32).reshape(-1, 3).astype(np.float64)
               if os.path.exists(p) else np.zeros((0, 3)))
        for ci, nm in enumerate(NAMES):
            pi = np.flatnonzero(np.asarray(cl) == ci)
            gi = [k for k, g in enumerate(G) if g['class'] == nm]
            if len(pi) == 0 or len(gi) == 0:
                data[(fid, nm)] = {
                    'pb': pb[pi], 'sc': np.asarray(sc, float)[pi],
                    'feat': np.zeros((len(pi), 5)),
                    'iou': (np.zeros(0), np.zeros(0, int),
                            np.zeros(0, int)),
                    'ngt': len(gi), 'npts': None}
                continue
            gb = np.array([G[k]['box'] for k in gi])
            # 门限：中心距 < 0.6*diag
            d = np.hypot(pb[pi][:, None, 0] - gb[None, :, 0],
                         pb[pi][:, None, 1] - gb[None, :, 1])
            diag = 0.5 * np.hypot(gb[:, 3], gb[:, 4]) + 0.5 * np.hypot(
                pb[pi][:, 3], pb[pi][:, 4])[:, None]
            m = d < diag * 1.05
            a, b = np.nonzero(m)
            ious = np.array([DE._iou_3d(gb[b[k]], pb[pi][a[k]]) for k in range(len(a))])
            keepm = ious > 0.05
            ft = np.array([box_feats(pts, q) for q in pb[pi]],
                          dtype=np.float64).reshape(-1, 4)
            ft = np.c_[ft, np.hypot(pb[pi][:, 0], pb[pi][:, 1])]
            data[(fid, nm)] = {'pb': pb[pi], 'sc': np.asarray(sc, float)[pi],
                               'feat': ft,
                               'iou': (a[keepm], b[keepm], ious[keepm]),
                               'ngt': len(gi), 'npts': None}
        print(f'  {fid} ok', flush=True)
    np.save(out, np.array(data, dtype=object))
    print('写出', out)


def _main_build():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='data/out/_rank_s12.npz')
    ap.add_argument('--clouds', default='data/eval_m4_holdout')
    ap.add_argument('--out', default='data/out/_ap1.npy')
    ap.add_argument('--build', action='store_true')
    a = ap.parse_args()
    if a.build or not os.path.exists(a.out):
        build(a.cache, a.clouds, a.out)


def ap40(scores, tps, n_gt):
    """官方 KITTI 40 点插值 AP（榜单口径）。"""
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


def greedy_tp(entry, w, thr=0.5):
    """给定每框权重 w（与 pb 同序），按 w 降序贪心匹配，返回 is_tp。"""
    a, b, v = entry['iou']
    npred = len(entry['sc'])
    tps = np.zeros(npred, bool)
    if len(a) == 0:
        return tps
    order = np.argsort(-w, kind='stable')
    adj = {}
    for aa, bb, vv in zip(a, b, v):
        if vv >= thr:
            adj.setdefault(int(aa), []).append((int(bb), float(vv)))
    used = np.zeros(entry['ngt'], bool)
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


def weight_of(wn, f):
    n = f[:, 0]
    if wn == 'base':
        return np.ones(len(n))
    if wn.startswith('d'):
        kk, rest = wn[1:].split('b')
        m = np.maximum(n * (f[:, 4] / 20.0) ** 2, 0.0)
        return (m / (m + float(kk))) ** float(rest)
    if wn.startswith('n'):
        kk, rest = wn[1:].split('b')
        return (n / (n + float(kk))) ** float(rest)
    return np.ones(len(n))


def split_validate(out, cands):
    """在偶数帧上挑参数、奇数帧上报告，检验增益是不是过拟合。"""
    data = np.load(out, allow_pickle=True).item()
    fids = sorted({k[0] for k in data})
    even = set(fids[::2])
    print(f'{"class":11s} {"n_gt":>5}  {"A:base":>7} {"A:best":>7} {"A:par":12s} '
          f'{"B:base":>7} {"B:best":>7}')
    for nm in NAMES:
        keys = [k for k in fids if (k, nm) in data]
        if not keys:
            continue
        ngt = sum(data[(k, nm)]['ngt'] for k in keys)

        def ev(wn, sub):
            S, T = [], []
            for k in sub:
                e = data[(k, nm)]
                if len(e['sc']) == 0:
                    continue
                w = weight_of(wn, e['feat'])
                s = e['sc'] * w
                S.append(s); T.append(greedy_tp(e, s))
            if not S:
                return 0.0
            nn = sum(data[(k, nm)]['ngt'] for k in sub)
            return ap11(np.concatenate(S), np.concatenate(T), nn)

        A = [k for k in keys if k in even]
        B = [k for k in keys if k not in even]
        ab = ev('base', A)
        best, par = ab, 'base'
        for wn in cands:
            v = ev(wn, A)
            if v > best:
                best, par = v, wn
        bb = ev('base', B)
        vb = ev(par, B)
        print(f'{nm:11s} {ngt:5d}  {ab:7.4f} {best:7.4f} {par:12s} {bb:7.4f} {vb:7.4f}')


CAND = (['n%gb%g' % (k, b) for k in (0, 2, 5, 10, 20, 50, 100, 200)
         for b in (0.5, 1.0, 1.5, 2.0)] +
        ['d%gb%g' % (k, b) for k in (2, 5, 10, 20, 50, 100)
         for b in (0.5, 1.0, 1.5, 2.0)])


def tune(out, cfg_path='reports/rerank_cfg.json'):
    """两折交叉：每类选"两半都涨且平均最高"的参数，写 JSON 供提交流水线用。"""
    data = np.load(out, allow_pickle=True).item()
    fids = sorted({k[0] for k in data})
    even = set(fids[::2])
    cfg = {}
    print(f'{"class":11s} {"n_gt":>5} {"A:base":>7} {"B:base":>7} '
          f'{"A:best":>7} {"B:best":>7}  params')
    base_all = []
    best_all = []
    for nm in NAMES:
        keys = [k for k in fids if (k, nm) in data]
        if not keys:
            continue
        ngt = sum(data[(k, nm)]['ngt'] for k in keys)

        def ev(wn, sub):
            S, T = [], []
            for k in sub:
                e = data[(k, nm)]
                if len(e['sc']) == 0:
                    continue
                s = e['sc'] * weight_of(wn, e['feat'])
                S.append(s); T.append(greedy_tp(e, s))
            if not S:
                return 0.0
            return ap40(np.concatenate(S), np.concatenate(T),
                        sum(data[(k, nm)]['ngt'] for k in sub))

        A = [k for k in keys if k in even]
        B = [k for k in keys if k not in even]
        ab, bb = ev('base', A), ev('base', B)
        base_all.append((ngt * ab + (ngt - ngt) * 0, ab, bb))
        best = None
        for wn in CAND:
            va, vb = ev(wn, A), ev(wn, B)
            if va <= ab and vb <= bb:
                continue
            if va < ab * 0.95 or vb < bb * 0.95:
                continue
            sc = 0.5 * (va + vb)
            if best is None or sc > best[0]:
                best = (sc, wn, va, vb)
        if best is None:
            print(f'{nm:11s} {ngt:5d} {ab:7.4f} {bb:7.4f} '
                  f'{"-":>7} {"-":>7}  保持默认')
            continue
        _, wn, va, vb = best
        mode = 'n' if wn[0] == 'n' else 'd'
        kk, rest = wn[1:].split('b')
        cfg[nm] = [mode, float(kk), float(rest)]
        print(f'{nm:11s} {ngt:5d} {ab:7.4f} {bb:7.4f} {va:7.4f} {vb:7.4f}  {wn}')
    import json
    json.dump(cfg, open(cfg_path, 'w'), indent=1)
    print('写出', cfg_path, cfg)


def search(out, weights):
    data = np.load(out, allow_pickle=True).item()
    fids = sorted({k[0] for k in data})
    print(f'{"class":11s} {"n_gt":>5} ' +
          '  '.join(f'{w:>8s}' for w in weights))
    for nm in NAMES:
        keys = [k for k in fids if (k, nm) in data]
        if not keys:
            continue
        ngt = sum(data[(k, nm)]['ngt'] for k in keys)
        line = [f'{nm:11s} {ngt:5d}']
        for wn in weights:
            S, T = [], []
            for k in keys:
                e = data[(k, nm)]
                if len(e['sc']) == 0:
                    continue
                f = e['feat']
                n = f[:, 0]
                w = np.ones(len(n))
                if wn == 'base':
                    pass
                elif wn.startswith('ne'):
                    kk, rest = wn[2:].split('b')
                    bb, gg = rest.split('g')
                    w = (n / (n + float(kk))) ** float(bb)
                    w = w * np.clip(f[:, 2], 0.05, 1.0) ** float(gg)
                elif wn.startswith('d'):
                    # 距离归一化点数：m = n * (r/20)^2，补偿点密度 ∝ 1/r²
                    kk, bb = wn[1:].split('b')
                    m = n * (f[:, 3] / 20.0) ** 2 if f.shape[1] > 3 else n
                    m = np.maximum(m, 0.0)
                    w = (m / (m + float(kk))) ** float(bb)
                elif wn.startswith('n'):
                    kk, bb = wn[1:].split('b')
                    w = (n / (n + float(kk))) ** float(bb)
                s = e['sc'] * w
                S.append(s)
                T.append(greedy_tp(e, s))
            ap = ap11(np.concatenate(S), np.concatenate(T), ngt)
            line.append(f'{ap:8.4f}')
        print('  '.join(line))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--tune':
        tune(sys.argv[2] if len(sys.argv) > 2 else 'data/out/_ap1.npy')
    elif len(sys.argv) > 1 and sys.argv[1] == '--split':
        split_validate(sys.argv[2] if len(sys.argv) > 2 else 'data/out/_ap1.npy',
                       ['n5b1', 'n20b1', 'n50b1', 'n100b1', 'n200b1', 'n20b2',
                        'd5b1', 'd10b1', 'd20b1', 'd50b1'])
    elif len(sys.argv) > 1 and sys.argv[1] == '--search':
        search(sys.argv[2] if len(sys.argv) > 2 else 'data/out/_ap1.npy',
               ['base', 'n5b1', 'n10b1', 'n20b1', 'n50b1', 'n5b2', 'n10b2',
                'n20b2', 'n100b1', 'n200b1',
                'd5b1', 'd10b1', 'd20b1', 'd50b1', 'd100b1'])
    else:
        _main_build()
