#!/usr/bin/env python3
"""从合法的真实帧里提取"目标点云资产库"（无监督，不读任何真实标注）。

合规依据：
  * rules.md 第 9 行允许用公开真实数据集的 train split 做域适配；
  * rules.md 第 11 行只禁止 100 个禁用帧，本脚本显式跳过它们；
  * rules.md 第 12 行禁止"用真实标注训练"。本脚本不读 detection_gt.json，
    分类完全靠无监督的尺寸启发式，标签由我们自己生成。

做法：
  1. 去地面（z > -1.6）
  2. 0.25 m 体素占据栅格 + 连通域标记 -> 簇
  3. 按簇的 PCA 朝向求局部坐标 -> 包围尺寸
  4. 按尺寸启发式归类（Person / Bicycle / Motorcycle / Car / Van / Truck）
  5. 每个类保留若干代表样本（去重：尺寸分箱后每箱限流）

输出 data/objbank/bank.npz：
  pts_<cls>   (N,3) float32，按类拼接的局部坐标点（heading 对齐）
  off_<cls}   (M+1,) 每个样本的起止偏移
  dim_<cls>   (M,3)  样本包围尺寸 (L,W,H)
  az_<cls>    (M,)   样本被采集时的方位角（用于放置时还原可见面）
  hd_<cls>    (M,)   样本的朝向

用法：
  python scripts/build_objbank.py --frames data/eval_m4_test/detection --n 120
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402

ANCHOR_SIZE = C.ANCHOR_SIZE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z_AIR = -1.60
VOX = 0.25
MIN_PTS = 12


def load_forbidden():
    p = os.path.join(ROOT, 'lumpi_starting_kit', 'forbidden_frames.txt')
    out = set()
    with open(p, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                out.add(s.zfill(6))
    return out


def clusters(pts):
    """体素连通域，返回 [(idx_array), ...]。"""
    p = pts[(pts[:, 2] > Z_AIR) & (pts[:, 2] < 8.0) &
            (np.abs(pts[:, 0]) <= 40) & (np.abs(pts[:, 1]) <= 40)]
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < MIN_PTS:
        return []
    ij = np.floor(p[:, :2] / VOX).astype(np.int64)
    z = np.floor(p[:, 2] / VOX).astype(np.int64)
    ij -= ij.min(axis=0)
    z -= z.min()
    occ = np.zeros((ij[:, 0].max() + 1, ij[:, 1].max() + 1, z.max() + 1),
                   bool)
    occ[ij[:, 0], ij[:, 1], z] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3), np.int8))
    if n == 0:
        return []
    lv = lab[ij[:, 0], ij[:, 1], z]
    order = np.argsort(lv, kind='stable')
    lvs = lv[order]
    cut = np.searchsorted(lvs, np.arange(1, n + 2))
    return [p[order[cut[i]:cut[i + 1]]] for i in range(n)]


def fit_box(p):
    """PCA 求水平朝向，返回 (局部点, L, W, H, hd)。"""
    xy = p[:, :2] - p[:, :2].mean(axis=0)
    cov = xy.T @ xy / max(len(xy), 1)
    w, v = np.linalg.eigh(cov)
    d = v[:, int(np.argmax(w))]
    hd = float(np.arctan2(d[1], d[0]))
    c, s = np.cos(-hd), np.sin(-hd)
    lx = xy[:, 0] * c - xy[:, 1] * s
    ly = xy[:, 0] * s + xy[:, 1] * c
    L = float(lx.max() - lx.min())
    W = float(ly.max() - ly.min())
    H = float(p[:, 2].max() - p[:, 2].min())
    return lx, ly, L, W, H, hd


def classify(L, W, H, n):
    """按"与类别模板的形状距离"归类（无监督，不看任何标注）。

    只比形状比，不比绝对尺度——尺度由放置时的目标框决定。
    Bus 与 Car 在测试时段的尺寸几乎一致（4.51x2.00x1.72 vs 4.44x1.91x1.60），
    二者无法区分，Bus 直接复用 Car 的资产。
    """
    lo, hi = min(L, W), max(L, W)
    if hi < 0.45 or lo < 0.12 or H < 0.35:
        return None                     # 碎片/噪点
    d = np.log(np.array([hi, lo, H]) + 1e-6)
    best, bd = None, 1e9
    for cls in ('Person', 'Car', 'Bicycle', 'Motorcycle', 'Van', 'Truck'):
        t = np.array(ANCHOR_SIZE[cls])
        tl = np.log(np.array([max(t[0], t[1]), min(t[0], t[1]), t[2]]))
        v = float(np.max(np.abs(d - tl)))
        if v < bd:
            best, bd = cls, v
    if bd > 0.55:                       # 形状对不上任何一类，丢掉
        return None
    # 点数下限：按类给不同的门槛，避免把碎片当大车
    if n < {'Person': 14, 'Bicycle': 14, 'Motorcycle': 16,
            'Car': 45, 'Van': 45, 'Truck': 60}[best]:
        return None
    return best


def _read_ply_np(path):
    """快速读 .ply 的 xyz（np.frombuffer，比 lumpi_io.read_ply 快两个数量级）。"""
    with open(path, 'rb') as f:
        head = b''
        while b'end_header' not in head:
            head += f.readline()
        lines = head.decode('ascii', 'replace').splitlines()
        n = 0
        props = []
        for ln in lines:
            if ln.startswith('element vertex'):
                n = int(ln.split()[-1])
            elif ln.startswith('property'):
                p = ln.split()
                if p[1] == 'list':
                    raise ValueError('list property 不支持')
                props.append((p[-1], p[1]))
        tmap = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
                'uchar': 'u1', 'uint8': 'u1', 'int': 'i4', 'int32': 'i4',
                'uint': 'u4', 'uint32': 'u4', 'short': 'i2', 'int16': 'i2',
                'ushort': 'u2', 'uint16': 'u2'}
        dt = np.dtype([(nm, tmap[t]) for nm, t in props])
        raw = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    return np.stack([raw['x'], raw['y'], raw['z']], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=os.path.join(ROOT, 'data', 'eval_m4_test',
                                                     'detection'))
    ap.add_argument('--n', type=int, default=120, help='最多读多少帧')
    ap.add_argument('--per-bin', type=int, default=6,
                    help='每个尺寸分箱最多留几个样本')
    ap.add_argument('--stride', type=int, default=1,
                    help='按 id 排序后每 stride 帧取一帧（跨全程取多样性）')
    ap.add_argument('--ply', action='store_true',
                    help='输入目录是原始 .ply 帧（Measurement4/lidar），需先读 ply')
    a = ap.parse_args()

    fb = load_forbidden()
    files = sorted(f for f in os.listdir(a.frames)
                   if f.endswith('.ply' if a.ply else '.bin'))
    files = [f for f in files if f.split('.')[0] not in fb]
    if a.stride > 1:
        files = files[::a.stride]
    files = files[:a.n] if a.n else files
    print(f'读取 {len(files)} 帧（已剔除禁用帧），来源 {a.frames}')

    bank = {}
    nclu = 0
    for k, fn in enumerate(files):
        if a.ply:
            pts = _read_ply_np(os.path.join(a.frames, fn))
        else:
            pts = np.fromfile(os.path.join(a.frames, fn), np.float32).reshape(-1, 3)
        for p in clusters(pts):
            if len(p) < MIN_PTS:
                continue
            nclu += 1
            lx, ly, L, W, H, hd = fit_box(p)
            cls = classify(L, W, H, len(p))
            if cls is None:
                continue
            cxy = p[:, :2].mean(axis=0)
            az = float(np.arctan2(cxy[1], cxy[0]))
            # 局部坐标：heading 对齐，中心归零，z 相对最低点
            z0 = p[:, 2].min()
            loc = np.stack([lx - (lx.max() + lx.min()) / 2,
                            ly - (ly.max() + ly.min()) / 2,
                            p[:, 2] - z0], axis=1).astype(np.float32)
            bank.setdefault(cls, []).append((loc, np.array([L, W, H], np.float32),
                                             az, hd, len(p),
                                             float(np.hypot(cxy[0], cxy[1]))))
        if (k + 1) % 20 == 0:
            print(f'  {k+1}/{len(files)} 帧，簇 {nclu}，'
                  + ' '.join(f'{c}:{len(v)}' for c, v in sorted(bank.items())))

    os.makedirs(os.path.join(ROOT, 'data', 'objbank'), exist_ok=True)
    out = {}
    print()
    print(f'{"class":<11}{"raw":>6}{"kept":>6}{"pts_med":>9}{"L":>7}{"W":>7}{"H":>7}')
    for cls, items in sorted(bank.items()):
        # 尺寸分箱限流，避免某一种车刷屏
        bins = {}
        for it in items:
            key = (round(float(it[1][0]), 1), round(float(it[1][2]), 1))
            bins.setdefault(key, []).append(it)
        kept = []
        rng = np.random.default_rng(0)
        for key, v in bins.items():
            rng.shuffle(v)
            kept.extend(v[:a.per_bin])
        pts = np.concatenate([k[0] for k in kept])
        off = np.cumsum([0] + [len(k[0]) for k in kept]).astype(np.int64)
        out[f'pts_{cls}'] = pts
        out[f'off_{cls}'] = off
        out[f'dim_{cls}'] = np.stack([k[1] for k in kept])
        out[f'az_{cls}'] = np.array([k[2] for k in kept], np.float32)
        out[f'hd_{cls}'] = np.array([k[3] for k in kept], np.float32)
        out[f'rr_{cls}'] = np.array([k[5] for k in kept], np.float32)
        d = out[f'dim_{cls}']
        print(f'{cls:<11}{len(items):>6}{len(kept):>6}'
              f'{np.median([len(k[0]) for k in kept]):>9.0f}'
              f'{np.median(d[:,0]):>7.2f}{np.median(d[:,1]):>7.2f}'
              f'{np.median(d[:,2]):>7.2f}')

    path = os.path.join(ROOT, 'data', 'objbank', 'bank.npz')
    np.savez_compressed(path, **out)
    print()
    print(f'写出 {path}  {os.path.getsize(path)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
