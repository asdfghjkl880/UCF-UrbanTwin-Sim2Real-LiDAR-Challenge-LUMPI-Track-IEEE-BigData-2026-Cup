"""把跨类发射应用到预测缓存并落盘，供 fastap2 反复扫描。"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pp_config as C  # noqa: E402
from launch_extra import gt_dims, post, VEH  # noqa: E402

NAMES = C.CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_sbg1_h.npz.npy"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/out/_sbg1_lx.npy"))
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--src-thr", type=float, default=0.25)
    ap.add_argument("--power", type=float, default=1.0, help="后验锐化指数")
    ap.add_argument("--hmin", type=float, default=0.0, help="仅从高于该高度的框发射")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    if not hasattr(z, "keys"):
        z = z.item()
    dims = gt_dims(a.gtdir)

    out = {}
    n0 = n1 = 0
    for fid, (b, s, c, g) in z.items():
        nb, ns, nc = [], [], []
        for i in range(len(b)):
            cls = NAMES[int(c[i])]
            nb.append(b[i]); ns.append(s[i]); nc.append(int(c[i]))
            n0 += 1
            if cls in VEH and float(s[i]) >= a.src_thr and float(b[i][5]) >= a.hmin:
                for tgt in VEH:
                    if tgt == cls:
                        continue
                    p = post(dims, tgt, float(b[i][3]), float(b[i][4]), float(b[i][5]))
                    if p < a.thr:
                        continue
                    sc = float(s[i]) * a.scale * min(1.0, p)
                    if sc < 0.02:
                        continue
                    nb.append(b[i]); ns.append(sc); nc.append(NAMES.index(tgt))
                    n1 += 1
        out[fid] = (np.asarray(nb, np.float32), np.asarray(ns, np.float32),
                    np.asarray(nc, np.int32), g)
    np.save(a.out, np.array(out, dtype=object), allow_pickle=True)
    print(f"原框 {n0}  +发射 {n1} = {n0+n1}  -> {a.out}.npy")


if __name__ == "__main__":
    main()
