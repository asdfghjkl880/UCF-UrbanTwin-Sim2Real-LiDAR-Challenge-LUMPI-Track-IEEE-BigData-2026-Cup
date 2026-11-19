"""用框内点云质心修正预测框中心（提升 3D IoU -> 提升最大召回）。

此前的接地消融显示：Person 只把中心换成 GT 时 AP 从 0.230 升到 0.688，
是所有几何误差里最要紧的一项。这里不改模型，只用框内点的均值给出一步修正：

    ctr' = ctr + t * R^T (mx, my),   z' = z + t * mz

其中 (mx,my,mz) 是点云质心在框坐标系下的偏移，R 由预测 yaw 给出。
t 在 A/B 两折上独立验证，只接受两折同时提升的 t；位移幅值做了裁剪，
避免把框甩到邻居目标上。

用法:
  py scripts/center_fix.py --npz data/out/_sbg3_final_rr.npz
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
import fastap2 as F  # noqa: E402
from rerank_pts import FI, box_feats  # noqa: E402

NAMES = C.CLASS_NAMES


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z if hasattr(z, "keys") else z.item()


def feats(z, clouds, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True).item()
        if d and len(next(iter(d.values()))) == len(FI):
            return d
    out = {}
    for n, fid in enumerate(sorted(z), 1):
        b = np.asarray(z[fid][0], np.float64)
        p = os.path.join(clouds, fid + ".bin")
        if not os.path.exists(p):
            out[fid] = np.zeros((len(b), len(FI)))
            continue
        q = np.fromfile(p, dtype=np.float32)
        pts = (q.reshape(-1, 4)[:, :3] if q.size % 4 == 0 and q.size % 3 != 0
               else q.reshape(-1, 3)).astype(np.float64)
        tree = cKDTree(pts)
        f = np.zeros((len(b), len(FI)))
        for i in range(len(b)):
            f[i] = box_feats(pts, tree, tuple(b[i][:7]))
        out[fid] = f
        if n % 20 == 0:
            print(f"  {n}/{len(z)}", flush=True)
    np.save(cache, out, allow_pickle=True)
    return out


def shift(z, fe, t, cmax=0.6):
    out = {}
    for fid, (b, s, c, g) in z.items():
        b = np.array(b, np.float64, copy=True)
        f = fe[fid]
        mx, my, mz = f[:, 15], f[:, 16], f[:, 17]
        co, si = np.cos(b[:, 6]), np.sin(b[:, 6])
        dx = co * mx - si * my
        dy = si * mx + co * my
        lim = cmax * np.maximum(b[:, 3], b[:, 4])
        nrm = np.hypot(dx, dy)
        sc = np.where(nrm > lim, lim / np.maximum(nrm, 1e-6), 1.0)
        b[:, 0] += t * dx * sc
        b[:, 1] += t * dy * sc
        b[:, 2] += t * np.clip(mz, -0.5 * b[:, 5], 0.5 * b[:, 5])
        out[fid] = (b, s, c, g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_sbg3_final_rr.npz"))
    ap.add_argument("--clouds", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "data/out/_cf.npy"))
    ap.add_argument("--cfg", default=os.path.join(ROOT, "reports/center_fix.json"))
    ap.add_argument("--ts", default="0,0.25,0.5,0.75,1.0")
    a = ap.parse_args()

    z = load_npz(a.npz)
    fe = feats(z, a.clouds, a.cache)
    fids = sorted(z)
    A = set(fids[::2])
    ts = [float(x) for x in a.ts.split(",")]

    base0, p0 = F.mean_ap(z)
    print(f"基线 mAP {base0:.4f}")

    best_t, gain = {}, {}
    print(f'{"class":11s} {"n_gt":>5} {"base":>7}  ' +
          "  ".join(f"t={t:<4}" for t in ts[:6]))
    for ci, nm in enumerate(NAMES):
        keys = [f for f in fids
                if any(int(r[0]) == ci for r in np.asarray(z[f][3]).reshape(-1, 8))]
        if not keys:
            continue

        def ap_sub(t, sub):
            zz = shift({f: z[f] for f in sub}, fe, t)
            return F.ap_one(zz, ci)[0]

        ab = ap_sub(0, [f for f in keys if f in A])
        bb = ap_sub(0, [f for f in keys if f not in A])
        line = [f"{nm:<11} {0:5d} {0.5*(ab+bb):7.4f}"]
        cand = []
        for t in ts:
            if t == 0:
                continue
            va = ap_sub(t, [f for f in keys if f in A])
            vb = ap_sub(t, [f for f in keys if f not in A])
            line.append(f"{(va+vb)/2:7.4f}")
            if va > ab and vb > bb:
                cand.append((0.5 * ((va - ab) + (vb - bb)), t, va, vb))
        cand.sort(reverse=True)
        t = cand[0][1] if cand else 0.0
        best_t[nm] = t
        gain[nm] = round(cand[0][0], 4) if cand else 0.0
        print("  ".join(line) + (f"   -> t={t} ({gain[nm]:+.4f})" if t else "   保持"))

    # 全类一起评估（每类用各自最优 t）
    zz = {}
    for fid in fids:
        b, s, c, g = z[fid]
        b = np.array(b, np.float64, copy=True)
        f = fe[fid]
        mx, my, mz = f[:, 15], f[:, 16], f[:, 17]
        co, si = np.cos(b[:, 6]), np.sin(b[:, 6])
        dx, dy = co * mx - si * my, si * mx + co * my
        lim = 0.6 * np.maximum(b[:, 3], b[:, 4])
        nrm = np.hypot(dx, dy)
        sc = np.where(nrm > lim, lim / np.maximum(nrm, 1e-6), 1.0)
        for i in range(len(b)):
            t = best_t.get(NAMES[int(c[i])], 0.0)
            if t:
                b[i, 0] += t * dx[i] * sc[i]
                b[i, 1] += t * dy[i] * sc[i]
                b[i, 2] += t * float(np.clip(mz[i], -0.5 * b[i, 5], 0.5 * b[i, 5]))
        zz[fid] = (b, s, c, g)
    m1, p1 = F.mean_ap(zz)
    print(f"\n全量 mAP  {base0:.4f} -> {m1:.4f}  ({m1-base0:+.4f})")
    for nm in NAMES:
        print(f"  {nm:<11}{p1[nm][0]:8.4f}{p0[nm][0]:8.4f}  ({p1[nm][0]-p0[nm][0]:+.4f})")
    json.dump(best_t, open(a.cfg, "w"), indent=1)
    print(f"-> {a.cfg}  {best_t}")
    if m1 > base0:
        np.save(a.npz.replace(".npz", "_cf.npz"), zz, allow_pickle=True)


if __name__ == "__main__":
    main()
