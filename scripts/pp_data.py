#!/usr/bin/env python3
"""
PointPillars 的数据管线：pillar 化、数据集、batch 拼接、anchor 编码。

全部用 numpy 实现，不依赖 spconv / 自定义 CUDA 算子，Windows 上直接跑。
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402

# 特征图上每个格子覆盖的实际米数。**必须由 FEAT_X 反推**：早先硬编码成
# 2*VOXEL_X=0.64，打开 PP_NECK=deep 后特征图变成 250x250，锚点却仍按 0.64m
# 铺开，整张锚点图被拉伸到 160m 并整体错位 —— 训练 loss 正常但真实 mAP 恒为 0。
STRIDE = (C.X_MAX - C.X_MIN) / C.FEAT_X  # 0.64m (shallow) / 0.32m (deep)
EPS = 1e-6


# ---------------------------------------------------------------- anchors
def build_anchors():
    """返回 (FEAT_Y, FEAT_X, NA, 7) 的 anchor 张量。

    排布刻意与卷积输出 (B, NA, H, W) 的 permute(0,2,3,1) 对齐，这样
    `cls.permute(0,2,3,1).reshape(B,-1)` 的扁平顺序与 ANCHOR_FLAT 完全一致。
    """
    tpl = C.anchor_array()                       # (NA, 7)
    xs = C.X_MIN + (np.arange(C.FEAT_X) + 0.5) * STRIDE
    ys = C.Y_MIN + (np.arange(C.FEAT_Y) + 0.5) * STRIDE
    a = np.repeat(tpl[None, None, :, :], C.FEAT_Y, axis=0)
    a = np.repeat(a, C.FEAT_X, axis=1).copy()    # (H, W, NA, 7)
    a[:, :, :, 0] = xs[None, :, None]
    a[:, :, :, 1] = ys[:, None, None]
    return a.astype(np.float32)


ANCHORS = build_anchors()                        # (H, W, NA, 7)
ANCHOR_CLS_GRID = np.broadcast_to(
    C.anchor_class_ids()[None, None, :],
    (C.FEAT_Y, C.FEAT_X, C.NA)).astype(np.int64)  # (H, W, NA)
ANCHOR_FLAT = ANCHORS.reshape(-1, 7)             # (H*W*NA, 7)
ANCHOR_CLS_FLAT = ANCHOR_CLS_GRID.reshape(-1)    # (H*W*NA,)
N_ANCHOR = ANCHOR_FLAT.shape[0]
# 高斯软标签用的 anchor 中心（flatten 后与 ANCHOR_CLS_FLAT 同序）
AX_F = np.ascontiguousarray(ANCHORS[..., 0].reshape(-1).astype(np.float32))
AY_F = np.ascontiguousarray(ANCHORS[..., 1].reshape(-1).astype(np.float32))
# 三维视图，供 bbox 局部切片（全图算 exp 会把每帧拖慢 10 倍）
AX_G = np.ascontiguousarray(ANCHORS[..., 0].astype(np.float32))   # (H,W,NA)
AY_G = np.ascontiguousarray(ANCHORS[..., 1].astype(np.float32))


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


# ---------------------------------------------------------------- pillarize
def pillarize(pts, max_pillars=None, max_pts=None):
    """(N,3) -> (feat (P,T,9), coords (P,2)=[iy,ix], npts (P,))"""
    max_pillars = max_pillars or C.MAX_PILLARS
    max_pts = max_pts or C.MAX_PTS_PER_PILLAR

    m = ((pts[:, 0] >= C.X_MIN) & (pts[:, 0] < C.X_MAX) &
         (pts[:, 1] >= C.Y_MIN) & (pts[:, 1] < C.Y_MAX) &
         (pts[:, 2] >= C.Z_MIN) & (pts[:, 2] < C.Z_MAX))
    p = pts[m]
    if len(p) == 0:
        return (np.zeros((0, max_pts, C.IN_FEATURES), np.float32),
                np.zeros((0, 2), np.int32), np.zeros(0, np.int32))

    ix = ((p[:, 0] - C.X_MIN) / C.VOXEL_X).astype(np.int32)
    iy = ((p[:, 1] - C.Y_MIN) / C.VOXEL_Y).astype(np.int32)
    np.clip(ix, 0, C.GRID_X - 1, out=ix)
    np.clip(iy, 0, C.GRID_Y - 1, out=iy)
    vid = ix * C.GRID_Y + iy

    uniq, inv, cnt = np.unique(vid, return_inverse=True, return_counts=True)
    # 点多的柱子优先保留（比"先到先得"更稳）
    rank = np.empty(len(uniq), np.int64)
    order = np.argsort(-cnt, kind='stable')
    rank[order] = np.arange(len(uniq))
    keep = rank < max_pillars

    pil = rank[inv]
    sel = pil < max_pillars
    p = p[sel]
    pil = pil[sel]
    n_pil = int(min(len(uniq), max_pillars))

    # 按柱子分组，组内保留前 max_pts 个点
    o = np.argsort(pil, kind='stable')
    p = p[o]
    pil = pil[o]
    start = np.searchsorted(pil, np.arange(n_pil), side='left')
    within = np.arange(len(pil)) - start[pil]
    ok = within < max_pts
    p = p[ok]
    pil = pil[ok]
    within = within[ok]
    npts = np.bincount(pil, minlength=n_pil).astype(np.int32)

    # 还原 (iy, ix)
    u = uniq[order[:n_pil]]
    kix = u // C.GRID_Y
    kiy = u % C.GRID_Y
    cx = C.X_MIN + (kix + 0.5) * C.VOXEL_X
    cy = C.Y_MIN + (kiy + 0.5) * C.VOXEL_Y
    cz = 0.5 * (C.Z_MIN + C.Z_MAX)

    feat = np.empty((len(p), C.IN_FEATURES), np.float32)
    feat[:, 0] = p[:, 0]
    feat[:, 1] = p[:, 1]
    feat[:, 2] = p[:, 2]
    feat[:, 3] = p[:, 0] - cx[pil]
    feat[:, 4] = p[:, 1] - cy[pil]
    feat[:, 5] = p[:, 2] - cz
    feat[:, 6] = cx[pil]
    feat[:, 7] = cy[pil]
    feat[:, 8] = np.hypot(p[:, 0], p[:, 1]) / 40.0

    out = np.zeros((n_pil, max_pts, C.IN_FEATURES), np.float32)
    out[pil, within] = feat
    coords = np.stack([kiy, kix], axis=1).astype(np.int32)
    return out, coords, npts


# ---------------------------------------------------------------- targets
def encode_targets(gt):
    """gt: (n,8) [cls, cx,cy,cz,L,W,H,hd] -> (pos_idx, reg_t, pos_cls, heat)

    pos_idx: 正样本 anchor 的扁平索引；reg_t: 回归目标 (P,7)；pos_cls: (P,)
    heat:    (N_ANCHOR,) float32 高斯软标签，峰值唯一（=1.0）在离框中心最近的
             同类 anchor 上，邻域按 exp(-d^2/2σ^2) 衰减，σ=0.5*min(L,W)。

    加 heat 的动机（实测）：旧实现用 POS_R 把中心邻域整块标成 1.0，推理时
    一个 1.6m×1.6m 的方框内约 25 个 anchor 分数几乎相同（平顶），峰值位置
    由噪声决定。Person 框仅 1.21m 对角，偏 1 格（0.32m）就把 3D IoU 打到
    0.34 < 0.5 —— 同一绝对误差下小目标全灭、Car（4.77m 对角）基本守住，
    正好对应实测的「Car 0.83→0.56，Person 0.83→0.016」。软标签让峰值唯一。
    """
    n = len(gt)
    pos_gt = np.full(N_ANCHOR, -1, np.int32)
    inside_any = np.zeros(N_ANCHOR, bool)
    if n == 0:
        return (np.zeros(0, np.int64), np.zeros((0, 7), np.float32),
                np.zeros(0, np.int64), np.zeros(N_ANCHOR, np.float32))

    # 小目标优先（同一 anchor 落在两个框里时给小目标，因为小目标更难）
    areas = gt[:, 4] * gt[:, 5]
    gt_order = np.argsort(areas)
    gt_has = np.zeros(n, bool)

    for gi in gt_order:
        cls = int(gt[gi, 0])
        cx, cy, cz, L, W, H, hd = gt[gi, 1:8]
        r = 0.5 * float(np.hypot(L, W)) + 0.8
        ix0 = max(0, int(np.floor((cx - r - C.X_MIN) / STRIDE)))
        ix1 = min(C.FEAT_X - 1, int(np.ceil((cx + r - C.X_MIN) / STRIDE)))
        iy0 = max(0, int(np.floor((cy - r - C.Y_MIN) / STRIDE)))
        iy1 = min(C.FEAT_Y - 1, int(np.ceil((cy + r - C.Y_MIN) / STRIDE)))
        if ix1 < ix0 or iy1 < iy0:
            continue
        sub = ANCHORS[iy0:iy1 + 1, ix0:ix1 + 1, :, :]        # (h, w, NA, 7)
        dx = sub[..., 0] - cx
        dy = sub[..., 1] - cy
        c, s = np.cos(hd), np.sin(hd)
        u = dx * c + dy * s
        v = -dx * s + dy * c
        inb = (np.abs(u) <= L / 2) & (np.abs(v) <= W / 2)     # (h, w, NA)
        if not inb.any():
            continue
        # 扁平索引：(iy, ix, anchor)
        yy, xx, aa = np.nonzero(inb)
        flat = (((yy + iy0) * C.FEAT_X + (xx + ix0)) * C.NA) + aa
        inside_any[flat] = True
        same = ANCHOR_CLS_GRID[0, 0, :][aa] == cls
        if not same.any():
            continue
        flat_s = flat[same]
        free = pos_gt[flat_s] < 0
        pos_gt[flat_s[free]] = gi
        gt_has[gi] = True

    # 兜底：Person(0.78x0.74) 在 0.64m 的网格上经常一个正 anchor 都分不到，
    # 这类目标就永远学不会、还全是漏检。给每个没分到的 GT 硬塞最近的同类 anchor。
    for gi in np.flatnonzero(~gt_has):
        cls = int(gt[gi, 0])
        cx, cy = gt[gi, 1], gt[gi, 2]
        ix = int(np.clip((cx - C.X_MIN) / STRIDE, 0, C.FEAT_X - 1))
        iy = int(np.clip((cy - C.Y_MIN) / STRIDE, 0, C.FEAT_Y - 1))
        # 同类别的两个朝向 anchor 都塞进去（朝向由回归决定）
        for a_ in range(C.NA):
            if ANCHOR_CLS_GRID[iy, ix, a_] != cls:
                continue
            f = (iy * C.FEAT_X + ix) * C.NA + a_
            if pos_gt[f] < 0:
                pos_gt[f] = gi
                gt_has[gi] = True

    # 小目标扩邻域。正样本数是"anchor 中心落在框内"决定的，正比于框面积：
    # 实测真实帧里 Car 有 47 个正 anchor/目标，Person 只有 3.2 个 —— 正样本率
    # 1.4e-5，焦点损失也压不过 21 万个负样本，Person/Bicycle/Unknown 的 AP
    # 恒为 0（而 Car 能到 0.3）。把"正样本不足"的目标的中心 3x3 邻域也标成正样本，
    # 让每个目标至少有 ~18 个正 anchor。推理时这些冗余框由 NMS 合并。
    per_gt = np.bincount(pos_gt[pos_gt >= 0], minlength=n)
    for gi in np.flatnonzero(per_gt < C.MIN_POS_PER_GT):
        cls = int(gt[gi, 0])
        ix0 = int(np.clip((gt[gi, 1] - C.X_MIN) / STRIDE, 0, C.FEAT_X - 1))
        iy0 = int(np.clip((gt[gi, 2] - C.Y_MIN) / STRIDE, 0, C.FEAT_Y - 1))
        for iy in range(max(0, iy0 - C.POS_R), min(C.FEAT_Y, iy0 + C.POS_R + 1)):
            for ix in range(max(0, ix0 - C.POS_R), min(C.FEAT_X, ix0 + C.POS_R + 1)):
                for a_ in range(C.NA):
                    if ANCHOR_CLS_GRID[iy, ix, a_] != cls:
                        continue
                    f = (iy * C.FEAT_X + ix) * C.NA + a_
                    if pos_gt[f] < 0:
                        pos_gt[f] = gi
                        gt_has[gi] = True

    pos_idx = np.flatnonzero(pos_gt >= 0)
    if len(pos_idx) > C.MAX_POS:
        pos_idx = np.random.choice(pos_idx, C.MAX_POS, replace=False)
        pos_idx.sort()

    gi = pos_gt[pos_idx]
    a = ANCHOR_FLAT[pos_idx]
    g = gt[gi]
    diag = np.hypot(a[:, 3], a[:, 4])
    reg = np.empty((len(pos_idx), 7), np.float32)
    reg[:, 0] = (g[:, 1] - a[:, 0]) / diag
    reg[:, 1] = (g[:, 2] - a[:, 1]) / diag
    reg[:, 2] = (g[:, 3] - a[:, 2]) / a[:, 5]
    reg[:, 3] = np.log(g[:, 4] / a[:, 3])
    reg[:, 4] = np.log(g[:, 5] / a[:, 4])
    reg[:, 5] = np.log(g[:, 6] / a[:, 5])
    reg[:, 6] = limit_period(g[:, 7] - a[:, 6])

    # ---- 高斯软标签：中心唯一峰值（bbox 局部计算，避免全图 exp）----
    heat = np.zeros(N_ANCHOR, np.float32)
    for gi_ in range(n):
        cls = int(gt[gi_, 0])
        cx, cy = float(gt[gi_, 1]), float(gt[gi_, 2])
        sig = float(np.clip(0.5 * min(float(gt[gi_, 4]), float(gt[gi_, 5])),
                            0.30, 2.0))
        r = 3.0 * sig
        ix0 = max(0, int(np.floor((cx - r - C.X_MIN) / STRIDE)))
        ix1 = min(C.FEAT_X - 1, int(np.ceil((cx + r - C.X_MIN) / STRIDE)))
        iy0 = max(0, int(np.floor((cy - r - C.Y_MIN) / STRIDE)))
        iy1 = min(C.FEAT_Y - 1, int(np.ceil((cy + r - C.Y_MIN) / STRIDE)))
        if ix1 < ix0 or iy1 < iy0:
            continue
        sx = AX_G[iy0:iy1 + 1, ix0:ix1 + 1, :]
        sy = AY_G[iy0:iy1 + 1, ix0:ix1 + 1, :]
        sc = ANCHOR_CLS_GRID[iy0:iy1 + 1, ix0:ix1 + 1, :]
        d2 = (sx - cx) ** 2 + (sy - cy) ** 2
        gau = np.exp(-d2 / (2.0 * sig * sig))
        same = sc == cls
        sel = same & (gau > 2e-3)
        if not sel.any():
            continue
        hh, ww, aa = np.nonzero(sel)
        flat = ((hh + iy0) * C.FEAT_X + (ww + ix0)) * C.NA + aa
        heat[flat] = np.maximum(heat[flat], np.minimum(gau[hh, ww, aa], 0.999))
        d2m = np.where(same, d2, np.inf)
        kh, kw, ka = np.unravel_index(int(np.argmin(d2m)), d2m.shape)
        heat[((kh + iy0) * C.FEAT_X + (kw + ix0)) * C.NA + ka] = 1.0

    return pos_idx.astype(np.int64), reg, g[:, 0].astype(np.int64), heat


# ---------------------------------------------------------------- dataset
def _augment(pts, gt, rng):
    """sim2real 增强。

    2026-11-10 追加"背景随机化"，这是最关键的一项。原因：8000 帧合成数据共用
    **同一张背景**（同一份静态地图、同一组扫描环、目标落在同一些车道上），模型
    于是把这张具体背景记了下来 —— 真实帧上每帧误报 500 个框、mAP 0.005。
    真实背景只要和它不一样，处处都像目标。

    对策是让背景在训练时"动起来"，逼模型只能依赖目标本身的形状：
      * 全局旋转 **±180°**（原来只 ±20°）：彻底打散"目标出现在画面哪个方位"
        这种记忆。目标框跟着转，几何仍自洽。
      * y 轴镜像 50%：等效于把整个路口镜像，视觉上是另一个场景。
      * 全局缩放 0.94~1.06：模拟安装高度/外参差异，框尺寸同步缩放。
      * 每样本随机丢点 U(0, 0.30)（原来 0.15）：真实回波数在同一目标上波动极大，
        固定丢点率会让模型把"点数多少"当成可靠线索。
      * 逐点噪声 1.5cm。
    """
    if rng.random() < 0.5:                       # 先镜像，再旋转
        pts = pts.copy()
        pts[:, 1] = -pts[:, 1]
        if len(gt):
            gt = gt.copy()
            gt[:, 2] = -gt[:, 2]
            gt[:, 7] = -gt[:, 7]

    ang = rng.uniform(-np.pi, np.pi)
    sc = float(rng.uniform(0.94, 1.06))
    # 平移量从 ±1m 放大到 ±3.5m：真实测试帧来自**行驶中的车**，自车在路口里的
    # 位置/朝向一直在变，而我们的合成场景自车永远停在原点。只靠 ±180° 旋转
    # 无法覆盖"目标相对自车的横向偏移"这一维，模型会把"目标出现在画面中央附近"
    # 当成先验。放大平移让整幅场景（背景 + 目标 + 框）一起挪，几何仍自洽。
    tx, ty = rng.uniform(-3.5, 3.5, 2)
    c, s = np.cos(ang), np.sin(ang)

    x = (pts[:, 0] * c - pts[:, 1] * s) * sc + tx
    y = (pts[:, 0] * s + pts[:, 1] * c) * sc + ty
    pts = np.stack([x, y, pts[:, 2] * sc], axis=1)
    pts = pts + rng.normal(0, 0.015, pts.shape).astype(np.float32)

    drop = float(rng.uniform(0.0, 0.30))
    if len(pts) > 100 and drop > 0:
        pts = pts[rng.random(len(pts)) >= drop]

    if len(gt):
        g = gt.copy()
        gx = (gt[:, 1] * c - gt[:, 2] * s) * sc + tx
        gy = (gt[:, 1] * s + gt[:, 2] * c) * sc + ty
        g[:, 1] = gx
        g[:, 2] = gy
        g[:, 3] = gt[:, 3] * sc
        g[:, 4:7] = gt[:, 4:7] * sc
        g[:, 7] = limit_period(gt[:, 7] + ang, 0.0, 2 * np.pi)
        gt = g
    return pts.astype(np.float32), gt


_MEM = {}


def _open_shard(stem):
    """按需 mmap 分片（mmap 对象不可 pickle，所以不能放在 Dataset 属性里）。"""
    if stem not in _MEM:
        _MEM[stem] = np.load(stem + '.npy', mmap_mode='r')
    return _MEM[stem]


class FrameDataset(Dataset):
    """支持两种存储：
      * 分片：shard_XXX.npy + shard_XXX_meta.json（合成集，避免海量小文件）
      * 单帧：<name>.bin + <name>.json（真实帧）
    """

    def __init__(self, dirs, augment=False, limit=None):
        if isinstance(dirs, str):
            dirs = [dirs]
        self.shards = []          # [{'stem','offs','labels'}]
        self.items = []           # (shard_idx, local_idx) 或 (-1, bin 路径前缀)
        for d in dirs:
            if not os.path.isdir(d):
                continue
            names = sorted(os.listdir(d))
            shard_names = [n for n in names
                           if n.startswith('shard_') and n.endswith('.npy')]
            if shard_names:
                for fn in shard_names:
                    stem = os.path.join(d, fn[:-4])
                    with open(stem + '_meta.json') as f:
                        meta = json.load(f)
                    k = len(self.shards)
                    self.shards.append({'stem': stem,
                                        'offs': np.asarray(meta['offsets'], np.int64),
                                        'labels': meta['labels']})
                    for i in range(len(meta['offsets']) - 1):
                        self.items.append((k, i))
            else:
                for fn in names:
                    if fn.endswith('.bin'):
                        self.items.append((-1, os.path.join(d, fn[:-4])))
        if limit:
            self.items = self.items[:limit]
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, ref = self.items[i]
        if k >= 0:
            sh = self.shards[k]
            a, b = int(sh['offs'][ref]), int(sh['offs'][ref + 1])
            pts = np.ascontiguousarray(_open_shard(sh['stem'])[a:b], dtype=np.float32)
            rows = sh['labels'][ref]
        else:
            pts = np.fromfile(ref + '.bin', dtype=np.float32).reshape(-1, 3)
            with open(ref + '.json') as f:
                rows = json.load(f)
        gt = np.array(rows, np.float32) if rows else np.zeros((0, 8), np.float32)
        if len(gt):
            # 只保留中心落在点云范围内的框：范围外的目标在裁剪后的点云里没有点，
            # 留着会变成学不到的"假负样本"
            keep = ((gt[:, 1] >= C.X_MIN) & (gt[:, 1] <= C.X_MAX) &
                    (gt[:, 2] >= C.Y_MIN) & (gt[:, 2] <= C.Y_MAX))
            gt = gt[keep]
        if self.augment:
            pts, gt = _augment(pts, gt, np.random.default_rng())
        feat, coords, npts = pillarize(pts)
        pos_idx, reg_t, pos_cls, heat = encode_targets(gt)
        return {
            'feat': torch.from_numpy(feat),
            'coords': torch.from_numpy(coords),
            'npts': torch.from_numpy(npts),
            'pos_idx': torch.from_numpy(pos_idx),
            'reg_t': torch.from_numpy(reg_t),
            'pos_cls': torch.from_numpy(pos_cls),
            'heat': torch.from_numpy(heat),
            'gt': torch.from_numpy(gt),
        }


def collate(batch):
    """把变长 pillar 补齐成 (B, P_max, T, 9)。"""
    P = max(b['feat'].shape[0] for b in batch)
    T = C.MAX_PTS_PER_PILLAR
    B = len(batch)
    feat = torch.zeros(B, P, T, C.IN_FEATURES)
    coords = torch.zeros(B, P, 2, dtype=torch.long)
    npts = torch.zeros(B, P, dtype=torch.long)
    pos_idx, reg_t, pos_cls, gts, heats = [], [], [], [], []
    for i, b in enumerate(batch):
        p = b['feat'].shape[0]
        if p:
            feat[i, :p] = b['feat']
            coords[i, :p] = b['coords'].long()
            npts[i, :p] = b['npts'].long()
        pos_idx.append(b['pos_idx'])
        reg_t.append(b['reg_t'])
        pos_cls.append(b['pos_cls'])
        gts.append(b['gt'])
        heats.append(b['heat'])
    return {
        'feat': feat, 'coords': coords, 'npts': npts,
        'pos_idx': pos_idx, 'reg_t': reg_t, 'pos_cls': pos_cls, 'gt': gts,
        'heat': heats,
    }


if __name__ == '__main__':
    import time
    print(f'anchors {N_ANCHOR:,}  ({C.NA} x {C.FEAT_Y} x {C.FEAT_X})')
    ds = FrameDataset([C.TRAIN_SYNTH], limit=20)
    t0 = time.time()
    stats = []
    for i in range(min(10, len(ds))):
        b = ds[i]
        stats.append((b['feat'].shape[0], int(b['npts'].float().mean().item()),
                      int(b['npts'].max().item()), len(b['pos_idx'])))
    print(f'每帧耗时 {(time.time()-t0)/min(10,len(ds)):.3f}s')
    print(f'{"pillars":>8}{"均点数":>8}{"最多点":>8}{"正样本":>8}')
    for s in stats:
        print(f'{s[0]:>8}{s[1]:>8}{s[2]:>8}{s[3]:>8}')
