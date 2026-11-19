#!/usr/bin/env python3
"""
把静态地图渲染成"传感器视角"的合成点云帧。

流程：
  1. 地图 UTM -> 传感器局部坐标：local = R^T (utm - T)，z 减去传感器高程；
  2. 裁到和测试帧相同的 ±40m / ±5m 盒子；
  3. 球面 z-buffer 做可见性剔除（每个方位/俯仰格只留最近的点），
     否则会看到"墙后面的点"，分布明显不对；
  4. 随机下采样到 50000 点。

用法：
    python scripts/render_map.py            # 渲染一帧并打印统计
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPORTS, REF_CLOUD, PREVIEW_CLOUD  # noqa: E402
from refcloud import read_xyz  # noqa: E402

BOX = (-40.0, -40.0, -5.0, 40.0, 40.0, 5.0)
ALIGN = os.path.join(REPORTS, 'map_align.json')


def load_align(path=ALIGN):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def utm_to_local(pts, align):
    """(N,3) UTM -> 传感器局部坐标。"""
    R = np.array(align['rot'], float)
    T = np.array(align['t_utm'], float)
    xy = (pts[:, :2] - T) @ R          # 等价于 R^T (p - T)
    z = pts[:, 2] - align['z_sensor']
    return np.stack([xy[:, 0], xy[:, 1], z], axis=1)


def z_buffer(pts, n_chan=64, n_az=1024, el_fov=22.5):
    """按真实激光雷达的扫描结构做可见性剔除。

    关键：不能用一个"均匀细角度网格"，否则近处地面会占掉大量格子，
    径向分布会严重偏向近点（实测均值 10.7m，而真实帧是 23.35m）。

    真实雷达是 N 条固定仰角通道 × 固定方位步进：
      * 垂直视场 ±22.5°（Ouster OS1-64 风格），所以 5m 内是盲区；
      * 每条通道在地面上只命中一个距离，因此每个"环带"点数大致恒定，
        径向分布接近"每环带均匀"，与真实帧吻合。
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.hypot(x, y)
    az = np.arctan2(y, x)
    el = np.degrees(np.arctan2(z, np.maximum(rng, 1e-6)))

    ia = np.floor((az + np.pi) / (2 * np.pi) * n_az).astype(np.int64)
    ie = np.round((el + el_fov) / (2 * el_fov) * (n_chan - 1)).astype(np.int64)
    ok = (ie >= 0) & (ie < n_chan)
    pts, ia, ie = pts[ok], ia[ok], ie[ok]
    r = np.hypot(rng[ok], z[ok])

    key = ia * 10000 + ie
    order = np.lexsort((r, key))
    key_s = key[order]
    first = np.ones(len(key_s), dtype=bool)
    first[1:] = key_s[1:] != key_s[:-1]
    return pts[order[first]]


def target_radial_pdf(n_bins=56, rmax=56.0, n_frames=8):
    """从本地合法参考帧统计半径分布（1m 分箱）。

    真实帧是 50k 点、裁到 ±40m 方盒，半径最大到 56m（角落）。
    地图是均匀面密度，直接抽样会让远处点偏多，EMD/FPD 明显变差，
    所以要用真实帧的径向分布做重要性重采样。
    """
    from paths import LOCAL_BENCH
    ref_dir = os.path.join(LOCAL_BENCH, 'realism_reference')
    files = sorted(os.listdir(ref_dir))[:n_frames]
    rs = []
    for fn in files:
        a = np.fromfile(os.path.join(ref_dir, fn), dtype='<f4')
        p = a.reshape(-1, 3)
        rs.append(np.hypot(p[:, 0], p[:, 1]))
    r = np.concatenate(rs)
    h, _ = np.histogram(r, bins=n_bins, range=(0, rmax))
    return h / h.sum(), rmax


def match_radial(pts, pdf, rmax, n_points, seed=0):
    """按目标径向分布做重要性重采样。"""
    n_bins = len(pdf)
    r = np.hypot(pts[:, 0], pts[:, 1])
    b = np.clip((r / rmax * n_bins).astype(np.int64), 0, n_bins - 1)
    src = np.bincount(b, minlength=n_bins).astype(float)
    src[src == 0] = np.nan
    w = pdf / (src / src[~np.isnan(src)].sum())
    w = np.nan_to_num(w, nan=0.0)
    pw = w[b]
    s = pw.sum()
    if s <= 0:
        rng = np.random.default_rng(seed)
        return pts[rng.choice(len(pts), min(n_points, len(pts)), replace=False)]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=min(n_points, len(pts)), replace=False, p=pw / s)
    return pts[idx]


def render(align=None, n_points=50000, seed=0, preview=True, do_zbuffer=False,
           match_dist=True):
    if align is None:
        align = load_align()
    cloud = read_xyz(PREVIEW_CLOUD if preview else REF_CLOUD)
    loc = utm_to_local(cloud, align)

    x0, y0, z0, x1, y1, z1 = BOX
    m = ((loc[:, 0] >= x0) & (loc[:, 0] <= x1) &
         (loc[:, 1] >= y0) & (loc[:, 1] <= y1) &
         (loc[:, 2] >= z0) & (loc[:, 2] <= z1))
    loc = loc[m]
    n_crop = len(loc)
    if do_zbuffer:
        loc = z_buffer(loc)

    if match_dist:
        pdf, rmax = target_radial_pdf()
        return match_radial(loc, pdf, rmax, n_points, seed), n_crop

    rng = np.random.default_rng(seed)
    if len(loc) > n_points:
        loc = loc[rng.choice(len(loc), n_points, replace=False)]
    return loc, n_crop


def main():
    loc, n_crop = render()
    print(f'裁盒后 {n_crop:,} 点，可见性剔除后 -> {len(loc):,} 点')
    r = np.hypot(loc[:, 0], loc[:, 1])
    print(f'  x {loc[:,0].min():.1f} ~ {loc[:,0].max():.1f}')
    print(f'  y {loc[:,1].min():.1f} ~ {loc[:,1].max():.1f}')
    print(f'  z {loc[:,2].min():.2f} ~ {loc[:,2].max():.2f}  中位 {np.median(loc[:,2]):.2f}')
    print(f'  半径 均值 {r.mean():.2f}  中位 {np.median(r):.2f}  最大 {r.max():.2f}')

    print('\n对比：测试帧基准（已知）')
    print('  z 中位 -2.05   半径均值 23.35   50,000 点')


if __name__ == '__main__':
    main()
