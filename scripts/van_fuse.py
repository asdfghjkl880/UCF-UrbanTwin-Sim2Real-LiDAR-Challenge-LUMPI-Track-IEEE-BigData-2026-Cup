"""Van 的分数融合：把模型原始 Van 置信度与跨类发射假设合并。

diag_van 揭示的事实：Van 的**召回其实有 0.659**（276 个 GT 命中 182），
但 AP 只有 0.083——分数排序把 TP 打散在 49,020 个预测里（top100 只有 27 个）。
所以瓶颈不是"学不会 Van"，而是**同一辆车的 Van 假设与 Car 假设各自带着
不同的分数**，谁也没帮上谁。

这里对每个发射候选（Car/Truck/Bus→Van）在同一位置找已有的 Van 预测，
把分数按几种规则融合：max / 加权和 / 概率 OR。目标是让真 Van 排到前面。

用法:
  py scripts/van_fuse.py --npz data/out/_sbg3_h.npz.npy --mode max --scale 0.5 --w 1.0
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
import train_pp as T  # noqa: E402
import fastap2 as F  # noqa: E402

NAMES = C.CLASS_NAMES
VEH = ["Car", "Van", "Truck", "Bus"]
VIDX = {v: NAMES.index(v) for v in VEH}
VI = VIDX["Van"]


def gt_dims(gtdir):
    gt = T.load_gt(gtdir)
    acc = {}
    for f in gt:
        for g in gt[f]:
            nm = g.get("class")
            if nm in NAMES:
                acc.setdefault(nm, []).append(g["box"][3:6])
    out = {}
    for nm, v in acc.items():
        v = np.asarray(v, np.float64)
        out[nm] = (v.mean(0), v.std(0) + 1e-3, v.min(0), v.max(0))
    return out


def post(dims, tgt, L, W, H):
    mu, sd, lo, hi = dims[tgt]
    p = 1.0
    for k, val in ((0, L), (2, H)):
        z = (val - mu[k]) / sd[k]
        p *= float(np.exp(-0.5 * z * z))
        if val < lo[k] - 0.2 or val > hi[k] + 0.2:
            p *= 1e-3
    return p


def build(z, dims, mode="max", scale=0.5, w=1.0, thr=0.02, src_thr=0.15,
          near=0.7):
    out = {}
    for fid, (b, s, c, g) in z.items():
        b = np.asarray(b, np.float64)
        s = np.asarray(s, np.float64).ravel()
        c = np.asarray(c).astype(np.int64).ravel()

        van_sel = np.flatnonzero(c == VI)
        vb = b[van_sel]
        vs = s[van_sel]

        nb, ns = [], []
        for i in np.flatnonzero(np.isin(c, [VIDX[v] for v in VEH]) & (s >= src_thr)):
            cls = NAMES[int(c[i])]
            if cls == "Van":
                continue
            p = post(dims, "Van", b[i, 3], b[i, 4], b[i, 5])
            if p < thr:
                continue
            sl = s[i] * scale * min(1.0, p)
            if len(vb):
                d = np.hypot(vb[:, 0] - b[i, 0], vb[:, 1] - b[i, 1])
                j = int(np.argmin(d))
                if d[j] < near:
                    sv = vs[j] * w
                    if mode == "max":
                        sl = max(sl, sv)
                    elif mode == "sum":
                        sl = sl + sv
                    elif mode == "or":
                        sl = 1.0 - (1.0 - min(sl, 1.0)) * (1.0 - min(sv, 1.0))
            if sl < 0.02:
                continue
            nb.append(b[i]); ns.append(sl)

        if nb:
            out[fid] = (np.vstack([b, np.asarray(nb)]),
                        np.concatenate([s, np.asarray(ns)]),
                        np.concatenate([c, np.full(len(nb), VI)]), g)
        else:
            out[fid] = (b, s, c, g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_sbg3_h.npz.npy"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--mode", default="max")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    if not hasattr(z, "keys"):
        z = z.item()
    dims = gt_dims(a.gtdir)
    base, pb = F.mean_ap(z)
    print(f"基线 mAP {base:.4f}  Van {pb['Van'][0]:.4f}\n")

    if not a.sweep:
        zz = build(z, dims, a.mode, a.scale, a.w)
        m, p = F.mean_ap(zz, verbose=True)
        print(f"\nmAP {m:.4f}  ({m-base:+.4f})  Van {p['Van'][0]:.4f}")
        return

    best = (base, None)
    for mode in ("max", "sum", "or"):
        for scale in (0.4, 0.5, 0.6):
            for w in (0.6, 1.0, 1.4):
                for near in (0.5, 0.7, 1.0):
                    zz = build(z, dims, mode, scale, w, near=near)
                    m, p = F.mean_ap(zz)
                    tag = ""
                    if m > best[0]:
                        best = (m, (mode, scale, w, near)); tag = " *"
                    print(f"  {mode:<4} sc={scale:<4} w={w:<4} near={near:<4}"
                          f" -> {m:.4f}  Van {p['Van'][0]:.4f}{tag}", flush=True)
    print(f"\n最佳 {best[1]}  mAP {best[0]:.4f}  ({best[0]-base:+.4f})")
    F.mean_ap(build(z, dims, *best[1]), verbose=True)


if __name__ == "__main__":
    main()
