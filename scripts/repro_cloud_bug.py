#!/usr/bin/env python3
"""故障复现：同一批预测、同一套后处理，只改"读哪一朵点云"，看 mAP 差多少。

在**有 GT** 的 133 个非禁用帧上跑两条路径：
  A 正确：apply_rows_full 用该帧真实点云 data/eval_m4_holdout/<fid>.bin
  B 线上：apply_rows_full 用随包合成帧 synthetic/000k.bin（当初打包脚本的 bug）
若 B 掉到官方线上量级（0.1169 附近）而 A 在本地的 0.2755 附近，则诊断闭环。

用法:
  py scripts/repro_cloud_bug.py --npz data/out/_f4h_final.npz.npy --frames 133
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
import fastap2 as F  # noqa: E402
from rerank_apply import apply_rows_full  # noqa: E402
from cloud_io import read_cloud  # noqa: E402


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z if hasattr(z, "keys") else z.item()


def to_rows(b, s, c):
    b = np.asarray(b, np.float64)
    s = np.asarray(s, np.float64).ravel()
    c = np.asarray(c).astype(np.int64).ravel()
    return [{"box": [float(v) for v in b[i]], "score": float(s[i]),
             "class": C.CLASS_NAMES[int(c[i])]} for i in range(len(b))]


def rows_to_npz(rows, g):
    if not rows:
        return (np.zeros((0, 7)), np.zeros(0), np.zeros(0, np.int64), g)
    b = np.array([r["box"] for r in rows], np.float64)
    s = np.array([r["score"] for r in rows], np.float64)
    c = np.array([C.CLASS_NAMES.index(r["class"]) for r in rows], np.int64)
    return (b, s, c, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_f4h_final.npz.npy"))
    ap.add_argument("--clouds", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--zip", default=os.path.join(ROOT, "data/out/submission_v23.zip"))
    ap.add_argument("--cfg", default=os.path.join(ROOT, "reports/rerank_pts.json"))
    ap.add_argument("--center-cfg", default=os.path.join(ROOT, "reports/center_fix.json"))
    ap.add_argument("--yaw-cfg", default=os.path.join(ROOT, "reports/yaw_fix.json"))
    ap.add_argument("--frames", type=int, default=133)
    a = ap.parse_args()

    Fz = load_npz(a.npz)
    ids = sorted(Fz)[: a.frames]
    cfg = json.load(open(a.cfg))
    tmap = json.load(open(a.center_cfg))
    ymap = json.load(open(a.yaw_cfg))

    with zipfile.ZipFile(a.zip) as zf:
        syn = [zf.read(n) for n in sorted(x for x in zf.namelist()
                                          if x.startswith("synthetic/"))]

    A, B = {}, {}
    for k, fid in enumerate(ids):
        b, s, c, g = Fz[fid]
        real = read_cloud(os.path.join(a.clouds, "%s.bin" % fid))
        r0 = to_rows(b, s, c)
        r1 = to_rows(b, s, c)
        apply_rows_full(r0, real, cfg, tmap, ymap)
        a0 = np.frombuffer(syn[k % len(syn)], dtype=np.float32)
        fake = (a0.reshape(-1, 4)[:, :3] if a0.size % 4 == 0 and a0.size % 3 != 0
                else a0.reshape(-1, 3)).astype(np.float64)
        apply_rows_full(r1, fake, cfg, tmap, ymap)
        A[fid] = rows_to_npz(sorted(r0, key=lambda x: -x["score"]), g)
        B[fid] = rows_to_npz(sorted(r1, key=lambda x: -x["score"]), g)

    out = []
    for tag, D in (("A 正确点云", A), ("B 合成点云(线上)", B)):
        m, p = F.mean_ap(D)
        out.append((tag, m, p))
        print("%s  mAP %.4f" % (tag, m))
        for nm in C.CLASS_NAMES:
            if nm in p:
                print("    %-11s %8.4f  (n_gt %d, n_pred %d)"
                      % (nm, p[nm][0], p[nm][1], p[nm][2]))

    print("\n官方线上实测 mAP 0.1169；本地正确路径基准 0.2755")
    for tag, m, _ in out:
        print("  %-16s %.4f   估算 combined %.4f"
              % (tag, m, 0.6 * m / 0.7 + 0.4 * 0.9307))


if __name__ == "__main__":
    main()
