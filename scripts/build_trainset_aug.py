#!/usr/bin/env python3
"""生成带 sim2real 增强的合成训练集（分片格式，与 build_trainset.py 一致）。

动机
----
实测域差距是 6.6 倍：合成域 mAP 0.622，真实域 0.094。且真实帧上模型疯狂误报
（Person 60 个 GT 输出 1684 框，合成域 120 个 GT 只输出 796 框）——说明它记住
的是**训练背景的具体结构**，真实背景一换就当成目标。

增强项（全部只作用于训练帧，提交帧保持不动，以免破坏 realism）：
  yaw      整帧绕 z 轴随机旋转（刚体变换，物理合法）—— 打散"结构在固定方位"
  drop     随机点丢弃，保留比例 p ~ U[lo, hi] —— 模拟不同反射率/距离的点数差异
  noise    测距方向高斯抖动
  flip     以 50% 概率沿 x 轴镜像（会翻转朝向，同步修正 heading）

用法:
    py scripts/build_trainset_aug.py --n 400 --shard 100 --workers 4 \
        --out data/train/aug0 --seed 900 --yaw 1 --drop 0.45,1.0 --noise 0.02
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402

AUG = dict(yaw=False, flip=False, drop=(1.0, 1.0), noise=0.0)


def _augment(pts, boxes, rng):
    pts = np.asarray(pts, np.float32).copy()
    bl = [np.asarray(b["box"], np.float64).copy() for b in boxes]

    if AUG["flip"] and rng.random() < 0.5:
        pts[:, 0] *= -1.0
        for b in bl:
            b[0] = -b[0]
            b[6] = np.pi - b[6]          # 镜像后朝向对称翻转
    if AUG["yaw"]:
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = x * c - y * s
        pts[:, 1] = x * s + y * c
        for b in bl:
            bx, by = b[0], b[1]
            b[0] = bx * c - by * s
            b[1] = bx * s + by * c
            b[6] = b[6] + th
    lo, hi = AUG["drop"]
    if hi < 1.0 or lo < 1.0:
        p = rng.uniform(lo, hi)
        if p < 1.0:
            n = max(64, int(len(pts) * p))
            idx = rng.choice(len(pts), n, replace=False)
            pts = pts[np.sort(idx)]
    if AUG["noise"] > 0:
        pts = pts + rng.normal(0, AUG["noise"], pts.shape).astype(np.float32)
    for b in bl:
        b[6] = (b[6] + np.pi) % (2 * np.pi) - np.pi
    return pts, bl


def _work(task):
    from scene_gen import gen_frame
    si, seeds = task
    rng = np.random.default_rng(seeds[0])
    chunks, offs, labs = [], [0], []
    for sd in seeds:
        pts, boxes = gen_frame(seed=sd)
        pts, bl = _augment(pts, boxes, rng)
        chunks.append(pts.astype(np.float32))
        offs.append(offs[-1] + len(pts))
        labs.append([[int(boxes[i]["class_id"])] +
                     [round(float(v), 3) for v in bl[i]] for i in range(len(boxes))])
    return si, np.vstack(chunks), offs, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yaw", type=int, default=1)
    ap.add_argument("--flip", type=int, default=1)
    ap.add_argument("--drop", default="0.45,1.0")
    ap.add_argument("--noise", type=float, default=0.02)
    a = ap.parse_args()

    AUG["yaw"] = bool(a.yaw)
    AUG["flip"] = bool(a.flip)
    AUG["drop"] = tuple(float(v) for v in a.drop.split(","))
    AUG["noise"] = a.noise

    os.makedirs(a.out, exist_ok=True)
    ns = (a.n + a.shard - 1) // a.shard
    tasks = [(s, [a.seed * 1000003 + s * a.shard + i
                  for i in range(min(a.shard, a.n - s * a.shard))])
             for s in range(ns)]
    t0 = time.time()
    done = 0
    with mp.Pool(a.workers) as pool:
        for si, pts, offs, labs in pool.imap_unordered(_work, tasks):
            stem = os.path.join(a.out, f"shard_{si:03d}")
            np.save(stem + ".npy", pts)
            with open(stem + "_meta.json", "w") as f:
                json.dump({"offsets": offs, "labels": labs}, f)
            done += 1
            el = time.time() - t0
            print(f"  片 {done}/{ns}  {len(pts):,} 点  {el:.0f}s  "
                  f"{done*a.shard/el:.1f} 帧/s", flush=True)
    print(f"完成 {a.n} 帧 -> {a.out}  {time.time()-t0:.0f}s  "
          f"aug yaw={AUG['yaw']} flip={AUG['flip']} drop={AUG['drop']} "
          f"noise={AUG['noise']}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
