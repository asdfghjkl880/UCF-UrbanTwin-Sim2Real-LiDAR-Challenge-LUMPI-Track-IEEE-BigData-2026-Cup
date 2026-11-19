#!/usr/bin/env python3
"""因果拆分：域差距来自「背景」还是「物体」？

构造两种混合帧，都带真实 GT，分别评估：
  A  realBG + synthOBJ   —— 真实背景（挖掉 GT 框内的点），用合成物体库在同样的框上
                            重新渲染目标
  B  synthBG + realOBJ   —— 合成背景 + 真实物体点（只保留 GT 框内的真实点）

判读：
  A 的 mAP 接近合成域的 0.62  → 背景是主因（真实背景能让模型正常工作）→ 那条路
                                可走：用真实帧背景承载合成物体
  B 的 mAP 接近 0.09          → 物体才是主因（真实物体点云模型认不出）

用法:
    py scripts/diag_split.py --real data/nog_holdout --n 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
import scene_gen as G  # noqa: E402
import train_pp as T  # noqa: E402

NAMES = C.CLASS_NAMES


def in_boxes(pts, boxes, m=0.20):
    keep = np.zeros(len(pts), bool)
    for b in boxes:
        cx, cy, cz, L, W, H, hd = [float(v) for v in b[:7]]
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        c, s = np.cos(hd), np.sin(hd)
        u = dx * c + dy * s
        v = -dx * s + dy * c
        keep |= ((np.abs(u) <= L / 2 + m) & (np.abs(v) <= W / 2 + m)
                 & (np.abs(pts[:, 2] - cz) <= H / 2 + m))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/nog_holdout")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out-a", default="data/cmp_Abg")
    ap.add_argument("--out-b", default="data/cmp_Bobj")
    a = ap.parse_args()

    os.makedirs(a.out_a, exist_ok=True)
    os.makedirs(a.out_b, exist_ok=True)
    gt = T.load_gt(a.real)
    fids = sorted(gt)[: a.n]
    bg = G.get_background()
    rng = np.random.default_rng(7)
    print(f"背景池 {len(bg):,}   帧 {len(fids)}")

    for k, fid in enumerate(fids):
        pts = np.fromfile(os.path.join(a.real, fid + ".bin"),
                          np.float32).reshape(-1, 3).astype(np.float64)
        boxes = np.array([g["box"] for g in gt[fid]], float)
        ins = in_boxes(pts, boxes)
        # A: 真实背景 + 合成物体
        bga = pts[~ins]
        objs = [(NAMES.index(g["class"]), *[float(v) for v in g["box"]])
                for g in gt[fid] if g["class"] in NAMES]
        if objs:
            op, _, _ = G.object_points(rng, objs)
        else:
            op = np.zeros((0, 3))
        pa = np.vstack([bga, op]) if len(op) else bga
        if len(pa) > 50000:
            pa = pa[rng.choice(len(pa), 50000, replace=False)]
        pa.astype(np.float32).tofile(os.path.join(a.out_a, fid + ".bin"))
        # B: 合成背景 + 真实物体
        pb = pts[ins]
        nb = max(0, 50000 - len(pb))
        if nb:
            idx = rng.choice(len(bg), nb, replace=False)
            pb = np.vstack([pb, bg[idx]])
        pb.astype(np.float32).tofile(os.path.join(a.out_b, fid + ".bin"))
        for d in (a.out_a, a.out_b):
            import shutil
            j = os.path.join(a.real, fid + ".json")
            if os.path.exists(j):
                shutil.copy(j, d)
        print(f"  {fid}  real_pts {len(pts)}  obj_in {int(ins.sum())}  "
              f"A {len(pa)}  B {len(pb)}", end="\r", flush=True)
    print("\n完成")


if __name__ == "__main__":
    main()
