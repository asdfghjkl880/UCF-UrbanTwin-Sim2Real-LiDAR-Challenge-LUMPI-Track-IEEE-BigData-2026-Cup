"""按类混合 TTA + 跨类发射的最终后处理。

TTA 对各类的效果是分化的（holdout 133 帧实测）：
  Person  0.158 -> 0.176  (+0.018)   <- 小目标，多视角平均显著降方差
  Car     0.661 -> 0.670  (+0.009)
  Bicycle 0.195 -> 0.160  (-0.035)   <- 镜像后朝向语义变了，反而有害
  Moto    0.364 -> 0.348  (-0.016)
  Van     0.083 -> 0.079  (-0.004)
  Truck   0.113 -> 0.113  ( 0.000)
所以只对 Person / Car 取 TTA 结果，其余类保留单次推理。

用法:
  py scripts/fuse_final.py --plain data/out/x_h.npz.npy --tta data/out/x_tta.npz.npy \
      --out data/out/x_final.npz
"""
import argparse
import os
import sys

import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pp_config as C  # noqa: E402
import fastap2 as F  # noqa: E402
from launch_extra import gt_dims, post, VEH  # noqa: E402
from rerank_apply import apply_npz  # noqa: E402

TTA_CLASSES = ("Person", "Car")
VEH_I = [C.CLASS_NAMES.index(v) for v in VEH]


def load(p):
    z = np.load(p, allow_pickle=True)
    return z if hasattr(z, "keys") else z.item()


def mix(P, T, tta_names=TTA_CLASSES):
    keep = [C.CLASS_NAMES.index(n) for n in tta_names]
    oth = [i for i in range(len(C.CLASS_NAMES)) if i not in keep]
    M = {}
    for fid in P:
        b1, s1, c1, g = P[fid]
        b2, s2, c2, _ = T[fid]
        c1 = np.asarray(c1).astype(np.int64)
        c2 = np.asarray(c2).astype(np.int64)
        m1 = np.isin(c1, oth)
        m2 = np.isin(c2, keep)
        M[fid] = (np.vstack([np.asarray(b1)[m1], np.asarray(b2)[m2]]),
                  np.concatenate([np.asarray(s1)[m1], np.asarray(s2)[m2]]),
                  np.concatenate([c1[m1], c2[m2]]), g)
    return M


def launch(z, dims, scale=0.5, thr=0.1, src_thr=0.1):
    out = {}
    for fid, (b, s, c, g) in z.items():
        b = np.asarray(b, np.float64)
        s = np.asarray(s, np.float64).ravel()
        c = np.asarray(c).astype(np.int64).ravel()
        nb, ns, nc = [], [], []
        for i in np.flatnonzero(np.isin(c, VEH_I) & (s >= src_thr)):
            cls = C.CLASS_NAMES[int(c[i])]
            for tgt in VEH:
                if tgt == cls:
                    continue
                p = post(dims, tgt, b[i, 3], b[i, 4], b[i, 5])
                if p < thr:
                    continue
                v = float(s[i]) * scale * min(1.0, p)
                if v < 0.02:
                    continue
                nb.append(b[i]); ns.append(v); nc.append(C.CLASS_NAMES.index(tgt))
        if nb:
            out[fid] = (np.vstack([b, np.asarray(nb)]),
                        np.concatenate([s, np.asarray(ns)]),
                        np.concatenate([c, np.asarray(nc)]), g)
        else:
            out[fid] = (b, s, c, g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default=os.path.join(ROOT, "data/out/_sbg3_h.npz.npy"))
    ap.add_argument("--tta", default=os.path.join(ROOT, "data/out/_sbg3_tta.npz.npy"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/out/_sbg3_final.npz"))
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--rerank-cfg",
                    default=os.path.join(ROOT, "reports/rerank_pts.json"))
    ap.add_argument("--clouds", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--center-cfg",
                    default=os.path.join(ROOT, "reports/center_fix.json"))
    ap.add_argument("--yaw-cfg", default=os.path.join(ROOT, "reports/yaw_fix.json"))
    ap.add_argument("--fused", default="",
                    help="已做 TTA 共识融合的 npz；给出时跳过 plain/tta 按类混合")
    a = ap.parse_args()

    dims = gt_dims(a.gtdir)

    if a.fused:
        P = load(a.fused if a.fused.endswith(".npy") else a.fused + ".npy")
        T = P
        m0, _ = F.mean_ap(P)
        M = P
        m1, p1 = F.mean_ap(M)
        print("使用 TTA 共识融合结果，跳过按类混合")
    else:
        P, T = load(a.plain), load(a.tta)
        m0, _ = F.mean_ap(P)
        M = mix(P, T)
        m1, p1 = F.mean_ap(M)
    L = launch(M, dims, scale=a.scale)
    m2, p2 = F.mean_ap(L)
    print(f"单次推理      {m0:.4f}")
    print(f"+ 类混合 TTA  {m1:.4f}  ({m1-m0:+.4f})")
    print(f"+ 跨类发射    {m2:.4f}  ({m2-m0:+.4f})")
    if os.path.exists(a.rerank_cfg):
        cfg = json.load(open(a.rerank_cfg))
        pts = {}
        for fid in L:
            p = os.path.join(a.clouds, fid + ".bin")
            if os.path.exists(p):
                q = np.fromfile(p, dtype=np.float32)
                pts[fid] = (q.reshape(-1, 4)[:, :3] if q.size % 4 == 0 and q.size % 3 != 0
                            else q.reshape(-1, 3)).astype(np.float64)
        tmap = (json.load(open(a.center_cfg))
                if os.path.exists(a.center_cfg) else {})
        ymap = (json.load(open(a.yaw_cfg)) if os.path.exists(a.yaw_cfg) else {})
        L = apply_npz(L, pts, cfg, tmap, ymap)
        print(f"  中心修正 { {k: v for k, v in tmap.items() if v} }")
        m3, p3 = F.mean_ap(L)
        print(f"+ 点云重排序  {m3:.4f}  ({m3-m0:+.4f})")
        p2 = p3
    for nm in C.CLASS_NAMES:
        print(f"  {nm:<11}{p2[nm][0]:8.4f}{p2[nm][1]:7d}{p2[nm][2]:8d}")
    if a.out:
        np.save(a.out, L, allow_pickle=True)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
