#!/usr/bin/env python3
"""验证：重排序用的是哪一朵点云，以及这决定性地改变了什么。

对同一批**未重排**的融合预测（data/out/_f4test_mix.npz.npy），分别用
  (A) 真实输入帧点云  lumpi_starting_kit/detection_test_frames/<fid>.bin
  (B) 随包合成帧      submission_v23.zip 里的 synthetic/000k.bin（当初的 bug）
计算 reports/rerank_pts.json 的逐类乘子 w，比较：
  1) 框内点数 n 的分布（真目标到底有没有被点云支撑）
  2) 乘子 w 的分布与"落到 0.01 保底"的比例
  3) 按 s*w 重排后每帧 top-40 与原始 top-40 的重合率（排序是否被换掉）

用法:
  py scripts/verify_rerank_cloud.py --npz data/out/_f4test_mix.npz.npy \
      --zip data/out/submission_v23.zip --frames 50 --per-class 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pp_config as C  # noqa: E402
from rerank_pts import box_feats, wfun  # noqa: E402
from cloud_io import read_cloud  # noqa: E402


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z if hasattr(z, "keys") else z.item()


def feats_for(rows, pts, tree=None):
    tree = tree if tree is not None else cKDTree(pts)
    out = np.zeros((len(rows), 19), np.float64)
    memo = {}
    for i, r in enumerate(rows):
        k = tuple(r["box"])
        v = memo.get(k)
        if v is None:
            v = np.asarray(box_feats(pts, tree, k), np.float64)
            memo[k] = v
        out[i] = v
    return out


def top_overlap(s, w, k=40):
    """原始 top-k 与加权后 top-k 的重合率（按类别内的行算）。"""
    n = len(s)
    if n == 0:
        return None
    k = min(k, n)
    a = set(np.argsort(-s)[:k].tolist())
    b = set(np.argsort(-(s * w))[:k].tolist())
    return len(a & b) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_f4test_mix.npz.npy"))
    ap.add_argument("--zip", default=os.path.join(ROOT, "data/out/submission_v23.zip"))
    ap.add_argument("--cfg", default=os.path.join(ROOT, "reports/rerank_pts.json"))
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--per-class", type=int, default=120)
    a = ap.parse_args()

    F = load_npz(a.npz)
    cfg = json.load(open(a.cfg))
    ids = sorted(F)[: a.frames]

    with zipfile.ZipFile(a.zip) as zf:
        syn = [zf.read(n) for n in sorted(x for x in zf.namelist()
                                          if x.startswith("synthetic/"))]

    stats = {}
    rng = np.random.RandomState(0)
    for k, fid in enumerate(ids):
        b, s, c, _g = F[fid]
        b = np.asarray(b, np.float64)
        s = np.asarray(s, np.float64).ravel()
        c = np.asarray(c).astype(np.int64).ravel()
        rows = [{"box": [float(v) for v in b[i]], "score": float(s[i]),
                 "class": C.CLASS_NAMES[int(c[i])]} for i in range(len(b))]

        real = read_cloud(os.path.join(C.TEST_FRAMES_DIR, "%s.bin" % fid))
        a0 = np.frombuffer(syn[k % len(syn)], dtype=np.float32)
        fake = (a0.reshape(-1, 4)[:, :3] if a0.size % 4 == 0 and a0.size % 3 != 0
                else a0.reshape(-1, 3)).astype(np.float64)

        tr, tf = cKDTree(real), cKDTree(fake)
        by = {}
        for i, r in enumerate(rows):
            by.setdefault(r["class"], []).append(i)

        for cls, idx in by.items():
            if len(idx) > a.per_class:
                idx = list(rng.choice(idx, a.per_class, replace=False))
            fr = feats_for([rows[i] for i in idx], real, tr)
            ff = feats_for([rows[i] for i in idx], fake, tf)
            wname = cfg.get(cls, "base")
            wr = wfun(wname, fr)
            wf = wfun(wname, ff)
            sc = s[idx]
            st = stats.setdefault(cls, {"n": [], "nr": [], "wr": [], "wf": [],
                                        "ov_r": [], "ov_f": []})
            st["n"].append(fr[:, 0])
            st["nr"].append(ff[:, 0])
            st["wr"].append(wr)
            st["wf"].append(wf)
            ovr = top_overlap(sc, wr)
            ovf = top_overlap(sc, wf)
            if ovr is not None:
                st["ov_r"].append(ovr)
                st["ov_f"].append(ovf)

    print("类          n_real  n_syn   w_real  w_syn  保底率real 保底率syn  top40重合real/syn")
    for cls in C.CLASS_NAMES:
        if cls not in stats:
            continue
        st = stats[cls]
        n_r = np.concatenate(st["n"])
        n_f = np.concatenate(st["nr"])
        w_r = np.concatenate(st["wr"])
        w_f = np.concatenate(st["wf"])
        print("%-11s %6.1f %6.1f %8.3f %7.3f %10.1f%% %9.1f%%   %.2f / %.2f"
              % (cls, np.median(n_r), np.median(n_f),
                 np.median(w_r), np.median(w_f),
                 100 * np.mean(w_r <= 0.02), 100 * np.mean(w_f <= 0.02),
                 np.mean(st["ov_r"]) if st["ov_r"] else -1,
                 np.mean(st["ov_f"]) if st["ov_f"] else -1))


if __name__ == "__main__":
    main()
