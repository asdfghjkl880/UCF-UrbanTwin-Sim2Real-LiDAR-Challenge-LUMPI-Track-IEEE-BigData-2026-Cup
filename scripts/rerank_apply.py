"""把 rerank_pts 找到的点云特征权重应用到提交行（json 行 或 npz 数组）。

训练/调参阶段的结论（holdout 133 帧，A/B 两折独立验证）:
  单因子  mAP 0.2313 -> 0.2501     三因子叠加 -> 0.2520
逐类：Person +0.017  Bicycle +0.038  Truck +0.035  Van +0.065

核心因子含义：
  n<k>b<b>     框内点数饱和权重 (n/(n+k))^b —— 点支撑
  d<k>b<b>     n*(r/20)^2 的距离归一化点数（补偿点密度 ∝ 1/r²）
  fill<b>      体积填充率 —— 框是否真的包住一个实体
  zr<b> / lr<b> / wr<b>   实测/预测 高、长、宽 比值 —— 遮挡完整度与尺寸自洽
  H<mu>x<sd> / L<mu>x<sd> 实测尺寸高斯 —— 类别形状先验（Truck/Van 靠这个区分）
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rerank_pts import FI, box_feats, wfun  # noqa: E402


def make_tree(pts):
    return cKDTree(pts) if len(pts) else None


def feats_for_rows(rows, pts, tree=None):
    """为一批行计算特征矩阵（按 box 元组去重，发射出的重复框共用一份）。"""
    if tree is None:
        tree = make_tree(pts)
    out = np.zeros((len(rows), len(FI)))
    memo = {}
    for i, r in enumerate(rows):
        key = (r["box"][0], r["box"][1], r["box"][2], r["box"][3],
               r["box"][4], r["box"][5], r["box"][6])
        v = memo.get(key)
        if v is None:
            v = box_feats(pts, tree, key) if tree is not None else [0.0] * len(FI)
            memo[key] = v
        out[i] = v
    return out


def apply_rows(rows, pts, cfg, floor_boost=0.0):
    """原地把 row['score'] 乘以该类权重，返回实际生效的行数。"""
    if not cfg:
        return 0
    tree = make_tree(pts)
    if tree is None:
        return 0
    fe = feats_for_rows(rows, pts, tree)
    n = 0
    for i, r in enumerate(rows):
        w = cfg.get(r["class"])
        if not w:
            continue
        r["score"] = float(r["score"]) * float(wfun(w, fe[i:i + 1])[0])
        n += 1
    return n


def apply_npz(z, pts_by_frame, cfg, tmap=None, ymap=None, cmax=0.6, amax=0.6):
    """npz 版（{fid: (boxes, scores, classes, gt)}）。tmap 非空时顺带做中心修正。"""
    out = {}
    for fid, (b, s, c, g) in z.items():
        s = np.array(s, np.float64, copy=True)
        pts = pts_by_frame.get(fid)
        if pts is not None and len(pts):
            tree = make_tree(pts)
            fe = np.zeros((len(b), len(FI)))
            memo = {}
            for i in range(len(b)):
                key = tuple(float(v) for v in np.asarray(b)[i][:7])
                v = memo.get(key)
                if v is None:
                    v = box_feats(pts, tree, key)
                    memo[key] = v
                fe[i] = v
            b = np.array(b, np.float64, copy=True)
            for i in range(len(b)):
                nm = _NAMES[int(c[i])]
                w = cfg.get(nm) if cfg else None
                if w:
                    s[i] *= float(wfun(w, fe[i:i + 1])[0])
                t = (tmap or {}).get(nm, 0.0)
                if not t:
                    continue
                mx, my, mz = fe[i, 15], fe[i, 16], fe[i, 17]
                L, W, H = b[i, 3], b[i, 4], b[i, 5]
                co, si = np.cos(b[i, 6]), np.sin(b[i, 6])
                dx, dy = co * mx - si * my, si * mx + co * my
                lim = cmax * max(L, W)
                nrm = float(np.hypot(dx, dy))
                if nrm > lim > 0:
                    dx, dy = dx * lim / nrm, dy * lim / nrm
                b[i, 0] += t * dx
                b[i, 1] += t * dy
                b[i, 2] += t * float(np.clip(mz, -0.5 * H, 0.5 * H))
            if ymap:
                for i in range(len(b)):
                    ty = ymap.get(_NAMES[int(c[i])], 0.0)
                    if ty:
                        b[i, 6] += ty * float(np.clip(fe[i, 18], -amax, amax))
        out[fid] = (np.asarray(b), s, np.asarray(c), g)
    return out


import pp_config as C  # noqa: E402

_NAMES = C.CLASS_NAMES


def apply_rows_full(rows, pts, cfg, tmap, ymap=None, cmax=0.6, amax=0.6):
    """一次特征计算，同时完成「分数重排序」与「点云质心中心修正」。

    tmap: {类别: t}，t=0 或缺失表示该类不做中心修正（Car/Truck/Van 实测
    修正有害，因为检测头本身在这些类上已经把中心par准了）。
    """
    if pts is None or not len(pts):
        return 0, 0
    tree = make_tree(pts)
    if tree is None:
        return 0, 0
    fe = feats_for_rows(rows, pts, tree)
    n_s = n_c = 0
    for i, r in enumerate(rows):
        w = cfg.get(r["class"]) if cfg else None
        if w:
            r["score"] = float(r["score"]) * float(wfun(w, fe[i:i + 1])[0])
            n_s += 1
        t = tmap.get(r["class"], 0.0) if tmap else 0.0
        if not t:
            continue
        mx, my, mz = fe[i, 15], fe[i, 16], fe[i, 17]
        if mx == 0.0 and my == 0.0 and mz == 0.0:
            continue
        L, W, H = r["box"][3], r["box"][4], r["box"][5]
        co, si = np.cos(r["box"][6]), np.sin(r["box"][6])
        dx, dy = co * mx - si * my, si * mx + co * my
        lim = cmax * max(L, W)
        nrm = float(np.hypot(dx, dy))
        if nrm > lim > 0:
            dx, dy = dx * lim / nrm, dy * lim / nrm
        r["box"][0] = float(r["box"][0] + t * dx)
        r["box"][1] = float(r["box"][1] + t * dy)
        r["box"][2] = float(r["box"][2] + t * float(np.clip(mz, -0.5 * H, 0.5 * H)))
        n_c += 1
    if ymap:
        for i, r in enumerate(rows):
            ty = ymap.get(r["class"], 0.0)
            if ty:
                r["box"][6] = float(r["box"][6]
                                    + ty * float(np.clip(fe[i, 18], -amax, amax)))
    return n_s, n_c
