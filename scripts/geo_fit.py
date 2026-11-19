#!/usr/bin/env python3
"""按类分治的几何精修 v3：多候选生成 + 无标签择优。

依据（60 合法真实帧，IoU@0.5 召回消融）：
    Person   0.230 -> 只修中心 0.688
    Bicycle  0.254 -> 只修朝向 0.570
    Motorcycle 0.195 -> 只修朝向 0.683
    Truck    0.158 -> 只修尺寸 0.335 / 中心 0.315
    Van      0.405 -> 只修中心 0.537
    Car      0.624 -> 只修中心 0.766

原则：**只生成该类真正敏感参数的候选**，再用"框内点数"择优，绝不允许
把点支撑变差的候选替换进来（v2 的教训：Car 0.62->0.40）。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

NAMES = ['Person', 'Car', 'Bicycle', 'Motorcycle', 'Bus', 'Truck', 'Van']

PRIOR = {
    'Person': (0.70, 0.70, 1.80), 'Car': (4.30, 1.80, 1.55),
    'Bicycle': (1.75, 0.65, 1.55), 'Motorcycle': (2.05, 0.75, 1.50),
    'Bus': (10.5, 2.55, 3.20), 'Truck': (6.60, 2.45, 2.85),
    'Van': (4.90, 2.00, 2.10),
}

# 每类允许改动的自由度
POLICY = {
    'Person':     dict(grid=2, ang=0, sz=0.0),   # 只修中心
    'Car':        dict(grid=0, ang=0, sz=0.0),   # 不动
    'Bicycle':    dict(grid=0, ang=6, sz=0.0),   # 只修朝向
    'Motorcycle': dict(grid=0, ang=6, sz=0.0),
    'Bus':        dict(grid=0, ang=0, sz=0.0),
    'Truck':      dict(grid=2, ang=0, sz=0.0),   # 中心
    'Van':        dict(grid=0, ang=0, sz=0.0),
}
GRID_STEP = 0.12


def _mask(pts, b, pad=1.0, zp=1.0):
    x, y, z, L, W, H, h = b
    c, s = np.cos(h), np.sin(h)
    px, py = pts[:, 0] - x, pts[:, 1] - y
    lx = c * px + s * py
    ly = -s * px + c * py
    m = (np.abs(lx) <= L / 2 * pad) & (np.abs(ly) <= W / 2 * pad)
    m &= (pts[:, 2] >= z - H / 2 * zp) & (pts[:, 2] <= z + H / 2 * zp)
    return m


def _ground(loc, b, r=1.6):
    m = ((loc[:, 0] - b[0]) ** 2 + (loc[:, 1] - b[1]) ** 2) < r * r
    if m.sum() < 10:
        return None
    return float(np.percentile(loc[m, 2], 8))


def _support(loc, b, zg, pad=0.85):
    m = _mask(loc, b, pad, 0.95)
    if zg is not None:
        m &= loc[:, 2] > zg + 0.25
    return int(m.sum())


def fit(loc, box, cls):
    """loc: 该框邻域候选点。返回精修框。"""
    b0 = np.asarray(box, np.float64).copy()
    pol = POLICY.get(cls)
    if pol is None or len(loc) < 10:
        return b0
    zg = _ground(loc, b0)
    s0 = _support(loc, b0, zg)

    cands = []
    if pol['grid'] > 0:
        n = pol['grid']
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                if i == 0 and j == 0:
                    continue
                c = b0.copy()
                c[0] += i * GRID_STEP
                c[1] += j * GRID_STEP
                cands.append(c)
    if pol['ang'] > 0:
        for k in range(pol['ang']):
            c = b0.copy()
            c[6] = b0[6] + (k + 0.5) * np.pi / pol['ang']
            cands.append(c)
    if not cands:
        return b0

    best = b0
    bs = s0
    for c in cands:
        s = _support(loc, c, zg)
        if s > bs:
            bs, best = s, c
    return best


def fit_frame(pts, boxes, classes, apply_to=None, tree=None, rad_pad=1.0):
    out = np.array(boxes, np.float64).copy()
    if tree is None:
        tree = cKDTree(pts)
    for i, b in enumerate(out):
        ci = int(classes[i])
        nm = NAMES[ci] if ci < len(NAMES) else 'Car'
        if apply_to is not None and nm not in apply_to:
            continue
        pol = POLICY.get(nm)
        if pol is None or (pol['grid'] == 0 and pol['ang'] == 0):
            continue
        r = 0.75 * float(np.hypot(b[3], b[4])) + 0.6 * float(b[5]) + rad_pad
        idx = tree.query_ball_point(b[:3], r)
        if len(idx) < 10:
            continue
        out[i] = fit(pts[idx], b, nm)
    return out
