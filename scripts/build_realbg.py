#!/usr/bin/env python3
"""用**真实帧背景**承载合成目标，生成训练集。

依据（diag_split.py，同 16 帧交叉实验）：
    真实背景 + 合成物体   mAP 0.1776
    真实背景 + 真实物体     0.1250
    合成背景 + 真实物体     0.1281
    合成背景 + 合成物体     0.6218
=> 背景是域差距的主因（0.622 -> 0.178），物体次之（0.178 -> 0.125）。

做法：从**非禁用**的真实帧里取点云，挖掉 GT 框内的点（只用于擦除真实物体，
不用它做任何监督），把得到的纯背景作为生成器的地基；物体位置、朝向、类别、
标签全部由我们自己的 scene_gen 决定。这样背景的环结构、多雷达、立面遮挡
100% 保真，而物体仍是合成目标库渲染。

两个子命令：
    --pool   从真实帧构建背景池 data/derived/realbg/bg_XXXXX.npy
    --gen    多进程生成分片训练集

用法:
    py scripts/build_realbg.py --pool --n-pool 600
    py scripts/build_realbg.py --gen --n 2000 --shard 100 --workers 10 --out data/train/rbg0
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
import train_pp as T  # noqa: E402

POOL = os.path.join(ROOT, "data", "derived", "realbg")
MARGIN = 0.25


def _erase(pts, boxes):
    keep = np.ones(len(pts), bool)
    for b in boxes:
        cx, cy, cz, L, W, H, hd = [float(v) for v in b[:7]]
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        c, s = np.cos(hd), np.sin(hd)
        u = dx * c + dy * s
        v = -dx * s + dy * c
        keep &= ~((np.abs(u) <= L / 2 + MARGIN) & (np.abs(v) <= W / 2 + MARGIN)
                  & (np.abs(pts[:, 2] - cz) <= H / 2 + MARGIN))
    return keep


def build_pool(src, n, out):
    os.makedirs(out, exist_ok=True)
    gt = T.load_gt(src)
    fids = [f for f in sorted(gt) if os.path.exists(os.path.join(src, f + ".bin"))]
    if n and len(fids) > n:
        idx = np.linspace(0, len(fids) - 1, n).astype(int)
        fids = [fids[i] for i in sorted(set(idx.tolist()))]
    ok = 0
    for fid in fids:
        pts = np.fromfile(os.path.join(src, fid + ".bin"),
                          np.float32).reshape(-1, 3)
        boxes = [g["box"] for g in gt[fid]]
        bg = pts[_erase(pts.astype(np.float64), boxes)]
        if len(bg) < 8000:
            continue
        bg.astype(np.float32).tofile(os.path.join(out, f"bg_{fid}.bin"))
        ok += 1
        print(f"  {ok}/{len(fids)}  {fid}  背景 {len(bg)} 点", end="\r", flush=True)
    print(f"\n背景池 {ok} 帧 -> {out}")


AUG = dict(yaw=True, flip=True, drop=(0.6, 1.0), noise=0.01,
           occ=0.35, wedge=True)


def _occlude_objects(pts, objs, rng, keep_lo=0.35, keep_hi=1.0, wedge=False):
    """模拟真实遮挡：框保持全尺寸，只随删掉框内的一部分点。

    真实标注的框是**物体完整外接框**，但激光常常只打到被遮挡后的一小片
    （实测 Car 有 21%、Truck 有 67% 的框内根本没有点）。合成里物体被完整
    粘贴（asset_points 的 drop 硬编码为 0），模型于是学成"点必须填满框"，
    迁到真实帧就把中心拽向可见点那一侧 —— 这正是真实帧上 0.25 m 中心误差
    和"小目标全崩、Car 幸存"的来源。
    """
    keep = np.ones(len(pts), bool)
    for o in objs:
        cx, cy, cz, L, W, H, hd = [float(v) for v in o[1:8]]
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        c, s = np.cos(hd), np.sin(hd)
        u = dx * c + dy * s
        v = -dx * s + dy * c
        m = ((np.abs(u) <= L / 2) & (np.abs(v) <= W / 2)
             & (np.abs(pts[:, 2] - cz) <= H / 2))
        idx = np.flatnonzero(m)
        if len(idx) < 3:
            continue
        if wedge and rng.random() < 0.5:
            # 整块切掉一半空间：更接近"只看到侧面"的真实情况
            fn = rng.normal(size=2)
            fn /= np.linalg.norm(fn) + 1e-9
            proj = u[idx] * fn[0] + v[idx] * fn[1]
            cut = idx[proj <= np.median(proj)]
            keep[cut] = False
            idx = idx[proj > np.median(proj)]
        p = rng.uniform(keep_lo, keep_hi)
        if p < 1.0 and len(idx) > 3:
            n = min(len(idx), max(3, int(len(idx) * p)))
            kept = rng.choice(idx, n, replace=False)
            drop = np.setdiff1d(idx, kept)
            keep[drop] = False
    return keep


def _aug(pts, objs, rng):
    pts = np.asarray(pts, np.float32).copy()
    bl = [np.asarray(o[1:8], np.float64) for o in objs]
    cls = [o[0] for o in objs]
    if AUG["flip"] and rng.random() < 0.5:
        pts[:, 0] *= -1
        for b in bl:
            b[0] = -b[0]
            b[6] = np.pi - b[6]
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
            b[6] += th
    lo, hi = AUG["drop"]
    p = rng.uniform(lo, hi)
    if p < 1.0:
        k = max(128, int(len(pts) * p))
        idx = rng.choice(len(pts), k, replace=False)
        pts = pts[np.sort(idx)]
    if AUG["noise"] > 0:
        pts = pts + rng.normal(0, AUG["noise"], pts.shape).astype(np.float32)
    for b in bl:
        b[6] = (b[6] + np.pi) % (2 * np.pi) - np.pi
    if AUG.get("occ", 0) > 0:
        objs_new = [[c] + list(b) for c, b in zip(cls, bl)]
        keep = _occlude_objects(pts, objs_new, rng,
                                keep_lo=AUG["occ"], keep_hi=1.0,
                                wedge=AUG.get("wedge", False))
        pts = pts[keep]
    return pts, [[int(c)] + [round(float(v), 3) for v in b]
                 for c, b in zip(cls, bl)]


def _work(task):
    global AUG
    from scene_gen import gen_frame
    si, seeds, pool, aug = task
    AUG = aug          # Windows spawn：子进程必须显式接收参数
    files = sorted(os.listdir(pool))
    rng = np.random.default_rng(seeds[0])
    chunks, offs, labs = [], [0], []
    for sd in seeds:
        fn = files[int(rng.integers(0, len(files)))]
        bg = np.fromfile(os.path.join(pool, fn), np.float32).reshape(-1, 3)
        pts, boxes = gen_frame(seed=sd, background=bg.astype(np.float64))
        objs = [(b["class_id"], *[float(v) for v in b["box"]]) for b in boxes]
        p2, lb = _aug(pts, objs, rng)
        chunks.append(p2.astype(np.float32))
        offs.append(offs[-1] + len(p2))
        labs.append(lb)
    return si, np.vstack(chunks), offs, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--src", default="data/train/m4_train")
    ap.add_argument("--n-pool", type=int, default=600)
    ap.add_argument("--dir-pool", default=POOL)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3100)
    ap.add_argument("--out", default="data/train/rbg0")
    ap.add_argument("--occ", type=float, default=0.35,
                    help="框内保留点的比例下限（模拟遮挡），1.0=关闭")
    ap.add_argument("--wedge", type=int, default=1)
    a = ap.parse_args()

    AUG["occ"] = a.occ
    AUG["wedge"] = bool(a.wedge)

    if a.pool:
        build_pool(a.src, a.n_pool, a.dir_pool)
    if a.gen:
        os.makedirs(a.out, exist_ok=True)
        ns = (a.n + a.shard - 1) // a.shard
        tasks = [(s, [a.seed * 1000003 + s * a.shard + i
                      for i in range(min(a.shard, a.n - s * a.shard))],
                  a.dir_pool, dict(AUG)) for s in range(ns)]
        t0 = time.time()
        done = 0
        with mp.Pool(a.workers) as p:
            for si, pts, offs, labs in p.imap_unordered(_work, tasks):
                stem = os.path.join(a.out, f"shard_{si:03d}")
                np.save(stem + ".npy", pts)
                with open(stem + "_meta.json", "w") as f:
                    json.dump({"offsets": offs, "labels": labs}, f)
                done += 1
                el = time.time() - t0
                print(f"  片 {done}/{ns}  {len(pts):,} 点  {el:.0f}s  "
                      f"{done*a.shard/el:.2f} 帧/s", flush=True)
        print(f"完成 {a.n} 帧 -> {a.out}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
