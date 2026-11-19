"""逐类选择：融合视角 vs 原始视角（两折各自独立判定）。

融合对各类的效果会分化（镜像 TTA 在 Bicycle/Moto 上本来就是负收益），
所以不能整类统一。这里对每一类在两折上分别比较「只用 p0」与「4 视角共识」，
只在该类两折都提升时才采用融合结果。
输出 reports/tta_pick.json: {类别: "p0" | "fused"}
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pp_config as C  # noqa: E402
import fastap2 as F  # noqa: E402
from tta_fuse import fuse_frame  # noqa: E402


def load(p):
    z = np.load(p, allow_pickle=True)
    return z if hasattr(z, "keys") else z.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", default=os.path.join(ROOT, "data/out/_p4h.npz"))
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--avg-dims", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports/tta_pick.json"))
    a = ap.parse_args()
    p = a.passes if a.passes.endswith(".npy") else a.passes + ".npy"
    z = load(p)
    fids = sorted(z)
    A = set(fids[::2])

    base, fused = {}, {}
    for f in fids:
        rec, gt = z[f]
        b, s, c = rec["p0"]
        base[f] = (np.asarray(b, np.float32), np.asarray(s, np.float32).ravel(),
                   np.asarray(c).astype(np.int64), gt)
        b2, s2, c2 = fuse_frame(rec, a.tau, a.alpha, a.gamma, bool(a.avg_dims))
        fused[f] = (b2, s2, c2, gt)

    pick = {}
    print(f"{'class':11s}{'A_p0':>8}{'A_fu':>8}{'':>3}{'B_p0':>8}{'B_fu':>8}{'':>3} 决定")
    for nm in C.CLASS_NAMES:
        ci = C.CLASS_NAMES.index(nm)
        zA0 = {f: base[f] for f in fids if f in A}
        zB0 = {f: base[f] for f in fids if f not in A}
        zA1 = {f: fused[f] for f in fids if f in A}
        zB1 = {f: fused[f] for f in fids if f not in A}
        a0 = F.ap_one(zA0, ci)[0]
        a1 = F.ap_one(zA1, ci)[0]
        b0 = F.ap_one(zB0, ci)[0]
        b1 = F.ap_one(zB1, ci)[0]
        ok = (a1 > a0) and (b1 > b0)
        pick[nm] = "fused" if ok else "p0"
        print(f"{nm:11s}{a0:8.4f}{a1:8.4f}{'':>3}{b0:8.4f}{b1:8.4f}{'':>3} "
              f"{'融合' if ok else '原样'}  ({a1-a0:+.4f}/{b1-b0:+.4f})")
    json.dump(pick, open(a.out, "w"), indent=1)
    print(f"-> {a.out}  采用融合的类: "
          f"{[k for k, v in pick.items() if v == 'fused']}")


if __name__ == "__main__":
    main()
