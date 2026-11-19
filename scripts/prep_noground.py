#!/usr/bin/env python3
"""去除地面点：训练集与推理端用**完全相同**的规则。

为什么这是当前最高杠杆的一步
--------------------------
实测：真实帧的地面是遍布全局的规则同心扫描环（5 台雷达 → 5 套环），而合成帧
的地面是"密度匹配的随机抽样"，没有环结构。于是模型学到的是"BEV 里密度稍高
的格子就是目标"，在真实帧上处处误报。
  * 合成域 mAP 0.622 / 真实域 0.094（6.6 倍）
  * 仅把真实帧的地面去掉（模型仍是在带地面的合成数据上训的），mAP 0.125 -> 0.139
两边同时去地面，环结构就从"域差距的来源"变成"两域都不存在的东西"。

地面高度不是常数（路面有坡度），所以按半径分箱取 z 的低分位数拟合 ground(r)，
再插值到每个点。

用法:
    py scripts/prep_noground.py --src data/train/s17p0 --out data/train/ng_s17p0
    py scripts/prep_noground.py --src data/eval_m4_holdout --out data/nog_holdout --bin
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RBIN = 3.0        # 半径分箱宽度
Q = 8.0           # 每箱取 z 的第 8 百分位作为地面
CLEAR = 0.35      # 地面以上此高度的点保留


def ground_curve(pts, rmax=60.0):
    r = np.hypot(pts[:, 0], pts[:, 1])
    edges = np.arange(0.0, rmax + RBIN, RBIN)
    idx = np.clip(np.digitize(r, edges) - 1, 0, len(edges) - 2)
    cen = 0.5 * (edges[:-1] + edges[1:])
    g = np.full(len(cen), np.nan)
    for k in range(len(cen)):
        m = idx == k
        if m.sum() >= 30:
            g[k] = np.percentile(pts[m, 2], Q)
    ok = np.isfinite(g)
    if ok.sum() == 0:
        return np.full(len(pts), np.percentile(pts[:, 2], Q))
    g = np.interp(cen, cen[ok], g[ok])
    return g[idx]


def keep_mask(pts):
    return pts[:, 2] > ground_curve(pts) + CLEAR


def shard(src, dst):
    os.makedirs(dst, exist_ok=True)
    n_in = n_out = 0
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".npy"):
            continue
        stem = fn[:-4]
        meta = json.load(open(os.path.join(src, stem + "_meta.json")))
        arr = np.load(os.path.join(src, fn), mmap_mode="r")
        offs = meta["offsets"]
        chunks, newoffs = [], [0]
        for i in range(len(offs) - 1):
            p = np.asarray(arr[offs[i]:offs[i + 1]], np.float32)
            p = p[keep_mask(p.astype(np.float64))]
            chunks.append(p)
            newoffs.append(newoffs[-1] + len(p))
            n_in += 1
        np.save(os.path.join(dst, fn), np.vstack(chunks))
        with open(os.path.join(dst, stem + "_meta.json"), "w") as f:
            json.dump({"offsets": newoffs, "labels": meta["labels"]}, f)
        n_out += 1
        print(f"  {stem}  帧位移 {n_in}  片 {n_out}", end="\r", flush=True)
    print(f"\n分片 {n_out} 个 -> {dst}")


def bindir(src, dst):
    os.makedirs(dst, exist_ok=True)
    tot = kept = 0
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".bin"):
            continue
        p = np.fromfile(os.path.join(src, fn), np.float32).reshape(-1, 3)
        q = p[keep_mask(p.astype(np.float64))]
        q.tofile(os.path.join(dst, fn))
        tot += len(p); kept += len(q)
        j = fn[:-4] + ".json"
        if os.path.exists(os.path.join(src, j)):
            shutil.copy(os.path.join(src, j), dst)
    print(f"帧文件 -> {dst}   点 {tot:,} -> {kept:,}（保留 {100*kept/max(tot,1):.0f}%）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", action="store_true", help="输入是 .bin+.json 目录")
    a = ap.parse_args()
    if a.bin:
        bindir(a.src, a.out)
    else:
        shard(a.src, a.out)


if __name__ == "__main__":
    main()
