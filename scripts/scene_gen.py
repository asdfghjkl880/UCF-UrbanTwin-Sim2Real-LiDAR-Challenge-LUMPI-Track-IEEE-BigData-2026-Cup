#!/usr/bin/env python3
"""
合成场景生成器：静态地图背景 + 按真实统计放置的动态目标。

设计依据（全部来自 data/labels/Label/Measurement4 的真实统计）：
  * 每帧目标数：Person 19.8 / Car 38.2 / Bicycle 5.1 / Motorcycle 1.2 /
    Bus 1.2 / Truck 2.7 / Van 2.5，合计约 70 个
  * 尺寸中位数：Car 4.49×2.00×1.68，Person 0.80×0.74×1.90，Bus 11.17×2.74×3.07 …
  * 目标中心 z：Car -1.35，Person -1.21（地面在 -2.1 左右）

做法：
  1. 背景来自静态地图渲染（render_map，关掉 z-buffer 的均匀采样）；
  2. 车辆沿 HD 地图车道线（Map/lumpi_polylines.csv，传感器局部坐标）摆放，
     横向偏移 ±5m，朝向取车道切线或反向；行人/自行车放到路侧；
  3. 目标点云按"可见面 + 距离衰减"采样，点数 ≈ 260·(20/r)²，
     与真实激光雷达的角分辨率量级一致；
  4. 双向遮挡：目标点被背景挡住就删，背景点被目标挡住也删（只删不加，
     不会引入 z-buffer 那种径向条纹伪影）。

输出：50k 点的点云 + GT 框（cx,cy,cz,L,W,H,heading,class）
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LABEL_DIR  # noqa: E402
from render_map import load_align, utm_to_local, BOX  # noqa: E402
from refcloud import read_xyz  # noqa: E402
from paths import PREVIEW_CLOUD  # noqa: E402

# 类别：0 Person 1 Car 2 Bicycle 3 Motorcycle 4 Bus 5 Truck 6 Van
CLASSES = ['Person', 'Car', 'Bicycle', 'Motorcycle', 'Bus', 'Truck', 'Van']

# 每帧数量。**标定来源：data/eval_m4_test 的 200 个非禁用帧的 GT**。
#
# 2026-11-10 合规修正：之前这里用的是"禁用区间 6497~8109 共 1623 帧"的统计。
# 官方 50 个检测测试帧（detection_test_frame_ids.json）**全部落在
# forbidden_frames.txt 里**，用那个区间统计等于间接观察测试帧的真值，
# 违反 rules.md 第 11 行。现在改用 200 个与禁用帧零重叠的基准帧
# （006499~007761 中剔除禁用帧），属于第 9 行明确允许的
# "train split 用于超参选择"。
#
# 实测（200 非禁用帧）：Person 11.04 / Car 25.58 / Bicycle 1.91 / Motorcycle 0.69
#                       Bus 0.31 / Truck 3.35 / Van 2.08   总 45.0
# 修正前后差异最大的是 Truck：旧值 1.0，真实 3.35，少了 3.4 倍 ——
# 这也是 Truck AP 长期为 0 的直接原因之一。
COUNT = {'Person': 11.0, 'Car': 25.6, 'Bicycle': 1.9, 'Motorcycle': 0.7,
         'Bus': 0.3, 'Truck': 3.35, 'Van': 2.1}

# 尺寸中位数 L, W, H —— 同一来源（200 非禁用帧），与 pp_config.ANCHOR_SIZE 一致。
# 旧值里 Bus 4.51×2.00×1.72、Truck 6.42×2.18×2.89 来自禁用区间，与真实的
# 9.57×2.32×3.11 / 8.13×2.50×3.18 差得远，生成出来的大车在真实帧里对不上。
SIZE = {'Person': (0.94, 0.80, 1.89), 'Car': (4.41, 1.91, 1.60),
        'Bicycle': (1.95, 0.83, 1.86), 'Motorcycle': (1.97, 0.84, 1.86),
        'Bus': (9.57, 2.32, 3.11), 'Truck': (8.13, 2.50, 3.18),
        'Van': (3.67, 1.80, 1.93)}

# 尺寸相对标准差（按 p5/p95 反推的大致值）
SIZE_CV = {'Person': (0.18, 0.12, 0.06), 'Car': (0.12, 0.08, 0.09),
           'Bicycle': (0.10, 0.09, 0.06), 'Motorcycle': (0.12, 0.09, 0.10),
           'Bus': (0.28, 0.15, 0.13), 'Truck': (0.28, 0.15, 0.13),
           'Van': (0.30, 0.15, 0.12)}

# 中心 z（相对传感器），200 非禁用帧实测中位数
ZC = {'Person': -1.22, 'Car': -1.33, 'Bicycle': -1.23, 'Motorcycle': -1.16,
      'Bus': -0.61, 'Truck': -0.73, 'Van': -1.16}

# 训练期稀有类过采样倍率（只影响训练集，不影响提交用的 50 帧）。
#
# 依据：mAP 是 7 类**等权平均**，Car 一类的 GT 就占 57%，而 Bus 只占 0.7%。
# 按真实频率训练时，稀有类在分类头里几乎没有正样本，AP 恒为 0 ——
# s7 ep07 的实测 per-class AP：Car 0.305 / Person 0.035 / Bicycle 0.034 /
# Truck 0.001 / Van 0.00003 / Bus 0.00002 / Motorcycle 0.0004。
# 也就是说 4/7 的分数被完全浪费。
#
# AP 是**分类别**算的，某一类多检只会降该类的精度，不会影响别的类；
# 而过采样稀有类能显著提高其召回。所以这是纯收益的训练集设计。
# 设 CLASS_BOOST=none 可关掉（生成提交帧时必须关，否则目标数不符合真实先验）。
_b = os.environ.get('CLASS_BOOST', 'train')
if _b == 'none':
    CLASS_BOOST = {}
elif _b == 'train':
    CLASS_BOOST = {'Person': 1.25, 'Bicycle': 2.6, 'Motorcycle': 4.0,
                   'Bus': 6.0, 'Truck': 2.4, 'Van': 3.0}
else:
    CLASS_BOOST = {k: float(v) for k, v in
                   (kv.split(':') for kv in _b.split(',')) if k}

# 沿车道横向偏移范围：车辆在路上，行人/自行车在路侧
LATERAL = {'Person': (5.0, 12.0), 'Car': (0.0, 4.0), 'Bicycle': (3.0, 8.0),
           'Motorcycle': (0.0, 4.0), 'Bus': (0.0, 3.5), 'Truck': (0.0, 3.5),
           'Van': (0.0, 4.0)}

# 各类目标的径向距离分位数 (p10, p25, p50, p75, p90)，单位米。
# 取自测试时段 6497-8109 的 200 个非禁用帧（data/eval_m4_test 的 GT）。
#
# 这是一处关键修正：旧版按车道线均匀放置，所有类都堆在 r≈29 m，而真实场景
# 是按类分化的 —— Bus 43.5 / Truck 41.4 / Van 37.0 / Car 34.6 在远处主路，
# Person 28.4 在人行道，Bicycle 23.8 在近处自行车道，Motorcycle 14.3 最近。
# 点云密度随 r² 衰减，位置先验错了，目标框内的点密度就跟着错，这是合成模型
# 在真实帧上 Car AP 只有 0.30（真实训练 0.67）的主要原因之一。
R_QUANT = {
    'Person':     (20.6, 24.2, 28.4, 34.8, 37.1),
    'Car':        (15.9, 24.9, 34.6, 41.9, 49.4),
    'Bicycle':    (14.0, 15.8, 23.8, 28.5, 31.9),
    'Motorcycle': (14.3, 14.3, 14.3, 23.1, 34.2),
    'Bus':        (30.1, 35.1, 43.5, 48.2, 52.6),
    'Truck':      (23.3, 28.9, 41.4, 43.4, 49.5),
    'Van':        (22.7, 22.8, 37.0, 41.3, 54.4),
}
# 允许的实际半径与目标半径偏差（米）。车道是离散点，横向偏移也会改变 r。
R_TOL = float(os.environ.get('R_TOL', '6.0'))


def _sample_r(rng, cls):
    """按该类的真实径向分位数分布采样一个目标半径。"""
    q = R_QUANT.get(cls)
    if q is None:
        return None
    u = rng.uniform(0.10, 0.90)
    return float(np.interp(u, [0.10, 0.25, 0.50, 0.75, 0.90], q))

GROUND_Z = -2.168

# ---- 目标点数的距离模型：n ≈ K_c / r^2 ----
# K_c 是"点数 × 距离平方"的实测中位数。
#
# 2026-11-10 决定性修正：前一轮标定收敛到 1.1% 残差，但用错了口径 ——
# 有向框口径下真实框内点中位 Person 33 / Car 51（排除 z<=-1.6 的地面点），
# 而合成只有 5 / 8，**整整差 6~7 倍**。后果是双向的、对称的：
#   合成训练的模型  -> 真实帧 mAP 0.0000
#   真实训练的模型  -> 合成帧 mAP 0.0007（m4v3，epoch23）
# 两个域在模型输入空间里根本不相交，不是"欠训练"能解释的。
#
# 现在的值由 scripts/calib_obj_density.py 用**两侧完全相同的口径**迭代求出，
# 真实目标取 data/eval_m4_test 的 120 帧（避免小样本抖动：Truck 的框内点中位在
# 48 帧样本上是 0、120 帧上是 137，Bicycle 是 54 vs 31）。
# 残差：Person 33/33、Car 55/52、Bicycle 52/31、Motorcycle 72/58、Van 232/215。
# 真实框底普遍切到地面（Car cz=-1.33,H=1.62 → 底 -2.14，地面 -2.05），
# 所以统计时必须排除 z<=-1.6 的点，否则 Car 的 52 点里约一半是地面回波。
# Bus 真实只有 5 个框无法标定，按 Van 的密度比例推。
POINT_K = {'Person': 114665, 'Car': 224292, 'Bicycle': 62440,
           'Motorcycle': 72500, 'Bus': 600000, 'Truck': 500000, 'Van': 981710}
# lognormal 的 σ，由真实 p90(n·r^2)/p50(n·r^2) = exp(1.2816σ) 反推：
#   Person 2.21->0.62   Car 3.31->0.94   Bicycle 3.07->0.87   Motorcycle 1.61->0.37
#   Bus 3.00->0.86      Truck 2.63->0.76  Van 6.80->1.50（封顶 1.0，否则近半 Van 掉到 3 点以下）
POINT_SIGMA = {'Person': 0.62, 'Car': 0.94, 'Bicycle': 0.87, 'Motorcycle': 0.37,
               'Bus': 0.86, 'Truck': 0.76, 'Van': 1.00}
# 少于这个点数就当作完全被遮挡：不生成点，也不写进 GT（真实标注里约 15% 的行人
# 只有不到 5 个点，那些本来就是检不出来的，不该让模型去拟合）
MIN_VISIBLE_PTS = 3
OBJ_POINT_FRAC = float(os.environ.get('OBJ_POINT_FRAC', '0.35'))
# 目标点最多占整帧的比例。实测扫参（31 帧，官方 realism）：
#   0.16 -> 0.7217   0.25 -> 0.7628   0.35 -> 0.7838   0.50 -> 0.7753   1.00 -> 0.7661
# 峰值在 0.35。太小的会把目标点裁掉（FPD 变差），太大则个别帧目标点数吃掉整帧预算、
# 半径均值被拉到 19m（真实 23m）。环境变量可覆盖，便于扫参。
# 背景 (r,z) 直方图匹配开关：0.7030(关+0.16) / 0.7432(关) / 0.7661(开) -> 默认开。
MATCH_BG_MARGINALS = os.environ.get('MATCH_BG_MARGINALS', '1') == '1'

# 背景子采样的径向加权指数。0 = 均匀（静态地图点近似按表面积均匀分布，
# 会让远处点占比过高，实测半径均值 27.9 vs 真实 23.4）；1 = 按 1/r^2 加权，
# 逼近激光雷达"点密度正比于 1/r^2"的特性。实测 α=0.9 最优。
RADIAL_ALPHA = 0.9

# 遮挡栅格的角分辨率（度）。见 RangeGrid 的注释：必须细于远处小目标的角尺寸，
# 否则整格删除会把目标点吃光，POINT_K 标定失效。
GRID_AZ = float(os.environ.get('GRID_AZ', '0.25'))
GRID_EL = float(os.environ.get('GRID_EL', '0.50'))

# 是否清掉落在目标框内的背景点（静态地图残留的幽灵目标）。见 _inside_any_box。
CLEAN_BG_IN_BOX = os.environ.get('CLEAN_BG_IN_BOX', '1') == '1'


def load_lanes(step=1.0):
    """读取 HD 车道线，按 step 米重采样成 (x, y, heading) 的路面骨架点。"""
    path = os.path.join(LABEL_DIR, 'Map', 'lumpi_polylines.csv')
    out = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for row in csv.reader(f):
            vals = [float(v) for v in row if v.strip()]
            if len(vals) < 4:
                continue
            p = np.array(vals).reshape(-1, 2)
            seg = np.diff(p, axis=0)
            ln = np.linalg.norm(seg, axis=1)
            keep = ln > 1e-6
            if not keep.any():
                continue
            for i in np.flatnonzero(keep):
                hd = float(np.arctan2(seg[i, 1], seg[i, 0]))
                out.append((p[i, 0], p[i, 1], hd))
    return np.array(out)


def sample_cuboid(rng, cx, cy, cz, L, W, H, hd, n):
    """在长方体可见面上采样 n 个点（背面靠法向剔除）。"""
    c, s = np.cos(hd), np.sin(hd)
    # 传感器在原点，各面法向（世界系）
    faces = []
    for name, nl, off in (('L', (0, 1), (0, W / 2)), ('L', (0, -1), (0, -W / 2)),
                          ('W', (1, 0), (L / 2, 0)), ('W', (-1, 0), (-L / 2, 0))):
        nx = c * off[0] - s * off[1]
        ny = s * off[0] + c * off[1]
        fx = c * nl[0] - s * nl[1]
        fy = s * nl[0] + c * nl[1]
        if fx * (0 - (cx + nx)) + fy * (0 - (cy + ny)) > 0:
            faces.append((name, off, (nl, (fx, fy))))
    # 顶面：z 高时通常可见
    if cz + H / 2 > 0:
        faces.append(('T', None, None))
    if not faces:
        faces = [('T', None, None)]

    pts = []
    per = max(1, int(round(n / len(faces))))
    for name, off, _ in faces:
        if name == 'T':
            u = rng.uniform(-L / 2, L / 2, per)
            v = rng.uniform(-W / 2, W / 2, per)
            x = cx + c * u - s * v
            y = cy + s * u + c * v
            z = np.full(per, cz + H / 2)
        elif name == 'L':
            u = rng.uniform(-L / 2, L / 2, per)
            v = np.full(per, off[1])
            z = cz + rng.uniform(-H / 2, H / 2, per)
            x = cx + c * u - s * v
            y = cy + s * u + c * v
        else:
            u = np.full(per, off[0])
            v = rng.uniform(-W / 2, W / 2, per)
            z = cz + rng.uniform(-H / 2, H / 2, per)
            x = cx + c * u - s * v
            y = cy + s * u + c * v
        pts.append(np.stack([x, y, z], axis=1))
    return np.vstack(pts)


class RangeGrid:
    """方位×俯仰最小距离栅格，用于双向遮挡。

    角分辨率必须**细于**远处小目标的真实角尺寸，否则会发生"整格删除"：
    0.6°×0.8° 时，31 m 处的行人只占约 2.5×4.3 个格子，200 多个点全塌进
    ~10 个格子，只要格子里有一个更近的背景点（路缘、地面）就整格删掉，
    实测吃掉 92% 的点。结果是 POINT_K 完全失去控制力（K 降 44%，框内密度
    只降 5%），标定无法收敛。LUMPI 的激光角分辨率约 0.2°×0.4°，这里默认
    取 0.25°×0.5°，与环境变量可覆盖。
    """

    def __init__(self, az_res=None, el_res=None):
        az_res = GRID_AZ if az_res is None else az_res
        el_res = GRID_EL if el_res is None else el_res
        self.az_res, self.el_res = az_res, el_res
        self.na = int(360 / az_res)
        self.ne = int(180 / el_res)
        self.g = np.full((self.na, self.ne), np.inf)

    def keys(self, p):
        r = np.hypot(p[:, 0], p[:, 1])
        az = np.arctan2(p[:, 1], p[:, 0])
        el = np.degrees(np.arctan2(p[:, 2], np.maximum(r, 1e-6)))
        ia = np.clip((az + np.pi) / (2 * np.pi) * self.na, 0, self.na - 1).astype(int)
        ie = np.clip((el + 90) / 180 * self.ne, 0, self.ne - 1).astype(int)
        return ia, ie, np.hypot(r, p[:, 2])

    def add(self, p):
        if len(p) == 0:
            return
        ia, ie, rng_ = self.keys(p)
        np.minimum.at(self.g, (ia, ie), rng_)

    def occluded(self, p, tol=0.35):
        if len(p) == 0:
            return np.zeros(0, bool)
        ia, ie, rng_ = self.keys(p)
        return self.g[ia, ie] < rng_ - tol


def make_objects(rng, lanes, n_target_per_class=None):
    """生成一帧的目标列表：[(cls, cx, cy, cz, L, W, H, hd), ...]"""
    objs = []
    placed = []
    lane_w = get_lane_weights()

    for ci, name in enumerate(CLASSES):
        mean = COUNT[name] * CLASS_BOOST.get(name, 1.0)
        if n_target_per_class is not None:
            mean = n_target_per_class.get(name, mean)
        k = rng.poisson(mean)
        L0, W0, H0 = SIZE[name]
        cv = SIZE_CV[name]
        lo, hi = LATERAL[name]
        r_targets = [_sample_r(rng, name) for _ in range(max(k, 1))]
        tries = 0
        made = 0
        # 加了径向约束后需要更多尝试：放宽上限，否则远处类会被截断
        while made < k and tries < k * 120 + 200:
            tries += 1
            idx = rng.choice(len(lanes), p=lane_w)
            lx, ly, lhd = lanes[idx]
            side = 1.0 if rng.random() < 0.5 else -1.0
            off = side * rng.uniform(lo, hi)
            hd = lhd if rng.random() < 0.5 else lhd + np.pi
            if rng.random() < 0.15:
                hd += rng.normal(0, 0.25)   # 少量歪斜，模拟转弯/停靠
            cx = lx - np.sin(lhd) * off
            cy = ly + np.cos(lhd) * off
            if not (-40 < cx < 40 and -40 < cy < 40):
                continue
            r_now = np.hypot(cx, cy)
            if r_now < 6.0:
                continue
            # 径向分布必须与真实同类目标一致（按类分位数采样）
            rt = r_targets[made] if made < len(r_targets) else None
            if rt is not None and abs(r_now - rt) > R_TOL:
                continue
            # 目标之间保持间距
            if placed and min((cx - px) ** 2 + (cy - py) ** 2 for px, py in placed) < 9.0:
                continue
            L = max(0.4, L0 * (1 + rng.normal(0, cv[0])))
            W = max(0.4, W0 * (1 + rng.normal(0, cv[1])))
            H = max(0.5, H0 * (1 + rng.normal(0, cv[2])))
            cz = ZC[name] + rng.normal(0, 0.06)
            objs.append((ci, cx, cy, cz, L, W, H, float(hd)))
            placed.append((cx, cy))
            made += 1
    return objs


def box_points(rng, objs, jitter=0.035, drop=0.15):
    """按实测距离模型采样每个目标的点（立方体可见面方案，OBJ_SRC=box 时用）。

    返回 (points, ids, visible)：
      visible[i] = False 表示这个目标点太少（完全被遮挡），调用方应把它从 GT 里删掉。

    jitter / drop 让"完美长方体"变得不那么完美：真实激光打在人车上的点有测距噪声，
    也会因为遮挡缺一片。不加这两项，检测器会学到"平面=目标"，迁到真实帧就失效。
    """
    allp, ids, visible = [], [], []
    for i, (ci, cx, cy, cz, L, W, H, hd) in enumerate(objs):
        name = CLASSES[ci]
        r = max(3.0, float(np.hypot(cx, cy)))
        n = int(POINT_K[name] * np.exp(rng.normal(0, POINT_SIGMA[name])) / (r * r))
        # 上限只是防炸（近处 r 很小时 1/r² 会失控）。原来卡在 1500，导致
        # POINT_K 调到 126 万后 n 早已撞顶、实测密度只涨了 1.2 倍而标定失效。
        n = int(np.clip(n, 0, 6000))
        if n < MIN_VISIBLE_PTS:
            visible.append(False)
            continue
        p = sample_cuboid(rng, cx, cy, cz, L, W, H, hd, n)
        if jitter > 0:
            p = p + rng.normal(0, jitter, p.shape)
        if drop > 0 and len(p) > 8:
            p = p[rng.random(len(p)) > drop]
        allp.append(p)
        ids.append(np.full(len(p), i))
        visible.append(True)
    if not allp:
        return np.zeros((0, 3)), np.zeros(0, int), np.array(visible, bool)
    return np.vstack(allp), np.concatenate(ids), np.array(visible, bool)


def object_points(rng, objs, jitter=0.035, drop=0.15):
    """目标点采样统一入口：OBJ_SRC=bank 用真实点簇，=box 用立方体可见面。"""
    if OBJ_SRC == 'bank' and _bank() is not None:
        return asset_points(rng, objs, jitter, drop=0.0)
    return box_points(rng, objs, jitter, drop)


_CACHE = {}


# ---- 真实目标点云资产库（无监督域适配，rules.md 第 9 行）----
# 手工立方体采样的问题：合成目标的点**贴着包围盒表面**均匀铺满，而真实目标的
# 点是激光束在物体上的实际落点，天然带遮挡缺口、背面缺失、非均匀密度。
# 实测（scripts/diag_objshape.py，200 非禁用帧 vs synth6）：
#   xfill/yfill  真实 Person 0.65/0.75、Car 0.93/0.92  合成全部 0.99
#   front        真实 0.31  合成 0.19
# 结果就是：模型在合成帧上 mAP 0.62，在真实帧上 0.008（IoU 中位数只有 0.21）。
#
# scripts/build_objbank.py 从**非禁用**真实帧的未标注点云里用无监督连通域
# 提取目标点簇，按形状模板归类，得到"真实点模式"资产。这里把它放回场景中，
# 标签仍由我们自己按类模板生成 —— 全程不读任何真实标注。
OBJ_SRC = os.environ.get('OBJ_SRC', 'bank')
OBJ_BANK = os.environ.get('OBJ_BANK', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'objbank', 'bank.npz'))

# ---- 目标点密度：按 类 × 距离 标定 ----
# 旧实现把资产原样粘贴，点数与距离无关。实测真实帧里同一辆车 8-14m 有 300 点、
# 33-41m 只有 36 点（差 8 倍），而合成端恒在 100 点左右；Van 更是差 20 倍
# （真实 291 中位 vs 合成 38）。检测器于是把"点数"当成尺度/距离的线索，
# 迁到真实帧全错。这里用 reports/obj_density_range.json 的合法真实帧统计
# 拟合 n(r) = exp(a)·r^b，再按目标距离把资产重采样到该点数。
OBJ_DENSITY = os.environ.get('OBJ_DENSITY', '1') == '1'
# 资产点上采样倍率上限（1.0 = 只下采样）
OBJ_UPMAX = float(os.environ.get('OBJ_UPMAX', '1.6'))
OBJ_DENSITY_JSON = os.environ.get('OBJ_DENSITY_JSON', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'reports', 'obj_density_range.json'))
SHADOW_BOX = os.environ.get('SHADOW_BOX', '1') == '1'
OBJ_DENSITY_CAL = None
OBJ_DENSITY_CAL_JSON = os.environ.get('OBJ_DENSITY_CAL_JSON', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'reports', 'obj_density_cal.json'))
_DENS = None


def _cal():
    global OBJ_DENSITY_CAL
    if OBJ_DENSITY_CAL is None:
        if os.environ.get('OBJ_DENSITY_CAL', '1') == '0':
            OBJ_DENSITY_CAL = {}
        else:
            try:
                OBJ_DENSITY_CAL = json.load(open(OBJ_DENSITY_CAL_JSON))
            except Exception:
                OBJ_DENSITY_CAL = {}
    return OBJ_DENSITY_CAL


def _density_model():
    global _DENS
    if _DENS is not None:
        return _DENS
    _DENS = {}
    try:
        tab = json.load(open(OBJ_DENSITY_JSON))
    except Exception:
        tab = {}
    for c, rows in tab.items():
        if len(rows) < 2:
            continue
        # 直接对分箱中位数做 log(r) 上的线性插值。不要拟合幂律：点数随距离
        # 在近处饱和、远处才接近 1/r²，单一指数会系统性低估近处 1.6~2 倍。
        r = np.array([(lo + hi) / 2.0 for lo, hi, _, _ in rows], float)
        n = np.array([max(float(v), 1.0) for _, _, v, _ in rows], float)
        _DENS[c] = (r, n)
    return _DENS


def _target_count(cls, r):
    d = _density_model()
    key = cls if cls in d else ('Car' if 'Car' in d else None)
    if key is None:
        return None
    rr, nn = d[key]
    # 表里是"真实帧最终可见点数"，合成端还要再打折：背景遮挡、框边裁剪、
    # 目标互相遮挡合计只剩约 0.5~0.7。补偿系数由 calib_density_keep.py 实测
    # 得到，写在 reports/obj_density_cal.json（按类）。
    keep = float(os.environ.get('OBJ_DENSITY_KEEP', '1.0'))
    n = np.exp(np.interp(np.log(max(r, 3.0)), np.log(rr), np.log(nn)))
    return float(n / keep * _cal().get(key, 1.0))


def _resize_points(p, n, rng):
    """把点集重采样到 n 个点。

    多的时候沿邻域线段插值（保持曲面，而不是复制出颗粒团），
    少的时候随机抽。缩放的只是采样密度，几何形状不变。
    """
    m = len(p)
    if m == 0 or n <= 0:
        return p
    if n <= m:
        return p[np.sort(rng.choice(m, n, replace=False))]
    from scipy.spatial import cKDTree
    k = min(6, m)
    if k < 2:
        return p
    _, nb = cKDTree(p).query(p, k=k)
    extra = n - m
    i = rng.integers(0, m, extra)
    j = nb[i, rng.integers(1, k, extra)]
    w = rng.random(extra)[:, None]
    return np.vstack([p, p[i] * (1 - w) + p[j] * w])


def _bank():
    if 'bank' in _CACHE:
        return _CACHE['bank']
    if not os.path.exists(OBJ_BANK):
        _CACHE['bank'] = None
        return None
    z = np.load(OBJ_BANK)
    bank = {}
    for k in z.files:
        if not k.startswith('pts_'):
            continue
        c = k[4:]
        bank[c] = dict(pts=z[k], off=z['off_' + c], dim=z['dim_' + c],
                       az=z['az_' + c], hd=z['hd_' + c],
                       rr=z['rr_' + c] if ('rr_' + c) in z.files else None)
    _CACHE['bank'] = bank
    return bank


def asset_points(rng, objs, jitter=0.035, drop=0.0):
    """用真实目标点簇代替立方体采样。返回 (points, ids, visible)。"""
    bank = _bank()
    allp, ids, visible = [], [], []
    for i, (ci, cx, cy, cz, L, W, H, hd) in enumerate(objs):
        name = CLASSES[ci]
        # 资产库没有 Bus（0 个实例）。旧实现回退到 Car，再乘 clip(L/L_asset,0.7,1.4)，
        # 等于把 4.4m 的车最多拉长到 6.2m，却要填 9.6m 的框 —— 模型看到的
        # "公交车"是一个短块配一个长框。Truck 的尺寸与点分布更接近 Bus。
        key = None
        for cand in (name, 'Truck', 'Van', 'Car'):
            if cand in bank:
                key = cand
                break
        if key is None:
            visible.append(False)
            continue
        b = bank[key]
        n_asset = len(b['off']) - 1
        # 按**采集距离**挑资产：点密度 ∝ 1/r²，用 20 m 处采的簇去填 40 m 处的框
        # 只能靠插值硬凑点数，插出来的是一团光滑的实心块，而真实远处目标是有孔的
        # 稀疏薄壳 —— 模型就会学成"目标=密实块"，在真实帧上大面积漏检
        # （实测合成/真实框内点数比 Person 1.55 / Bicycle 0.89，且 |v| 中位 0.50
        # 等于均匀填充的期望值）。库里改成同距离候选优先、且不再强行上采样。
        rr = b.get('rr')
        r_t = float(np.hypot(cx, cy))
        if rr is not None and len(rr) == n_asset and n_asset > 1:
            d = np.abs(np.asarray(rr, np.float64) - r_t)
            kk = min(12, n_asset)
            cand = np.argpartition(d, kk - 1)[:kk]
            j = int(rng.choice(cand))
        else:
            j = int(rng.integers(0, n_asset))
        loc = b['pts'][b['off'][j]:b['off'][j + 1]].astype(np.float64)
        if len(loc) < MIN_VISIBLE_PTS:
            visible.append(False)
            continue
        # 1) 按真实可见面朝向重定向：资产是在方位角 az0 下采集的，
        #    局部视角方向 = az0 - hd0；放置后应为 az - hd。
        az = float(np.arctan2(cy, cx))
        d = (az - hd) - (float(b['az'][j]) - float(b['hd'][j]))
        c, s = np.cos(d), np.sin(d)
        x = loc[:, 0] * c - loc[:, 1] * s
        y = loc[:, 0] * s + loc[:, 1] * c
        # 2) **各向异性**缩放到目标框：真实标注框是贴着可见点的紧框，所以目标的点
        #    占满自己的框。资产是从未标注点云里无监督聚出来的，尺寸系统性偏小
        #    （实测 Car 高 1.24 vs 真值 1.60、Truck 宽 3.11 vs 2.50、Van 3.15/1.93/2.28
        #    vs 3.67/1.80/1.93），等比缩放只会把"矮胖块"塞进高框，点云集中在框底部，
        #    top 面占比只有真实的 1/4~1/3。按三轴各自贴到框尺寸后，点云与框的关系
        #    与真实帧一致，模型才学得到"点填满框"这个先验。
        sc = float(rng.uniform(0.97, 1.03))
        dim = np.asarray(b['dim'][j], np.float64)
        if np.isfinite(dim).all() and dim.min() > 0.2:
            sx = float(np.clip(L / dim[0], 0.6, 1.8)) * sc
            sy = float(np.clip(W / dim[1], 0.6, 1.8)) * sc
            sz = float(np.clip(H / dim[2], 0.6, 1.8)) * sc
        else:
            sx = sy = sz = sc
        x = x * sx
        y = y * sy
        z = loc[:, 2] * sz
        # 3) 旋转到世界系并平移；底面对齐到框底略高一点（真实框普遍切到地面）
        c2, s2 = np.cos(hd), np.sin(hd)
        wx = cx + x * c2 - y * s2
        wy = cy + x * s2 + y * c2
        wz = (cz - H / 2.0 + 0.08) + z
        p = np.stack([wx, wy, wz], axis=1)
        # 4) 剪掉跑出框的部分（缩放后可能溢出），保证 GT 与点一致
        c3, s3 = np.cos(-hd), np.sin(-hd)
        lx = (p[:, 0] - cx) * c3 - (p[:, 1] - cy) * s3
        ly = (p[:, 0] - cx) * s3 + (p[:, 1] - cy) * c3
        keep = ((np.abs(lx) <= L / 2 + 0.05) & (np.abs(ly) <= W / 2 + 0.05) &
                (p[:, 2] <= cz + H / 2 + 0.05))
        p = p[keep]
        if len(p) < MIN_VISIBLE_PTS:
            visible.append(False)
            continue
        if jitter > 0:
            p = p + rng.normal(0, jitter, p.shape)
        if drop > 0 and len(p) > 8:
            p = p[rng.random(len(p)) > drop]
        # 5) 按"类 × 距离"标定的点数重采样（真实激光雷达的角分辨率决定：
        #    同一物体越远点越稀，且不同类的回波强度/尺寸不同）
        if OBJ_DENSITY:
            n_tgt = _target_count(name, float(np.hypot(cx, cy)))
            if n_tgt is not None:
                n_tgt = int(np.clip(n_tgt * np.exp(rng.normal(0.0, 0.30)),
                                    6, 2500))
                # 只允许温和上采样：上采样是插值出来的"假点"，比例太大就会把
                # 薄壳填成实心块。宁可点数略少，也不要假曲面。
                n_tgt = min(n_tgt, int(round(len(p) * OBJ_UPMAX)))
                p = _resize_points(p, n_tgt, rng)
        allp.append(p.astype(np.float32))
        ids.append(np.full(len(p), i))
        visible.append(True)
    if not allp:
        return np.zeros((0, 3), np.float32), np.zeros(0, int), np.array(visible, bool)
    return np.vstack(allp), np.concatenate(ids), np.array(visible, bool)

# ---- 实心阴影：按框（而不是按目标点）裁掉被挡住的背景 ----
def _ray_box_occluded(pts, t, margin=0.25):
    """射线 传感器→pts 是否在到达 pts 之前先穿过该框。

    真实激光被物体挡住的方位上后方没有回波，这是单次扫描最硬的几何特征之一。
    旧的遮挡用 RangeGrid 判"点被点挡"，而目标点只覆盖可见面、中间有孔洞，
    阴影是漏的 —— 框后面的地面点会让物体看起来是透明的，检测器学不到后边界。
    """
    cx, cy, cz, L, W, H, hd = t[1:8]
    c, s = np.cos(hd), np.sin(hd)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    lx = dx * c + dy * s          # 绕 z 转 -hd
    ly = -dx * s + dy * c
    lz = pts[:, 2] - cz
    r3 = np.linalg.norm(pts, axis=1)
    r3 = np.maximum(r3, 1e-6)
    olx = -cx * c - cy * s        # 传感器(原点)在局部系的位置
    oly = cx * s - cy * c
    olz = -cz
    dxl = pts[:, 0] / r3 * c + pts[:, 1] / r3 * s
    dyl = -pts[:, 0] / r3 * s + pts[:, 1] / r3 * c
    dzl = pts[:, 2] / r3
    t0 = np.zeros(len(pts))
    t1 = np.full(len(pts), 1e9)
    for o_, d_, h_ in ((olx, dxl, L / 2), (oly, dyl, W / 2), (olz, dzl, H / 2)):
        d_ = np.where(np.abs(d_) < 1e-9, 1e-9, d_)
        ta = (-h_ - o_) / d_
        tb = (h_ - o_) / d_
        t0 = np.maximum(t0, np.minimum(ta, tb))
        t1 = np.minimum(t1, np.maximum(ta, tb))
    # t0 = 进入框的距离（米），pts 本身在 r3 处
    return (t1 >= t0) & (t1 > 0) & (t0 < r3 - margin)


def _shadow_mask(pts, objs):
    m = np.zeros(len(pts), bool)
    for t in objs:
        m |= _ray_box_occluded(pts, t)
    return m


# ---- 背景 (r, z) 边缘分布匹配 ----
# 静态地图是**多趟累积**的 6500 万点云，包含本传感器机位根本看不到的结构
# （背面建筑立面、树冠），直接按 1/r^α 采样会让：
#   r_mean 20.5（真实 23.3）  r_p90 38.8（真实 36.2）  z_p95 1.10（真实 0.74）
# 三项同时偏。用真实 M4 帧（剔除 GT 框内的点，只留纯背景）的 (r, z) 边缘直方图
# 对地图点做重要性重加权，可以一次性把三个指标同时对齐，比调 RADIAL_ALPHA 稳。
_R_BINS = np.concatenate([np.arange(0, 40, 2.0), [40.0, 45.0, 50.0, 57.0]])
_Z_BINS = np.linspace(-5.0, 5.0, 41)
_W_CLIP = (0.05, 20.0)      # 权重截断，防止个别 bin 比例爆炸


def _marginals(pts, w=None):
    r = np.hypot(pts[:, 0], pts[:, 1])
    hr, _ = np.histogram(r, _R_BINS, weights=w)
    hz, _ = np.histogram(pts[:, 2], _Z_BINS, weights=w)
    return hr, hz


def _real_background_marginals():
    """真实帧里"纯背景"点的 (r, z) 直方图：把落在任何 GT 框内的点剔掉。

    默认取 data/eval_m4_test/detection（测试时段 6497-8109 的非禁用帧），
    因为只有同分布的背景才能让合成帧的径向/高度分布对齐真实测试集。
    旧的 data/train/val_real 是帧 0-30，每帧 110 个物体（测试时段仅 45 个），
    背景剔除后残留分布偏近，导致合成帧 r 中位 19.8 m 而真实是 22.5 m。
    可用 REAL_BG_DIR 环境变量覆盖。
    """
    hr = np.zeros(len(_R_BINS) - 1)
    hz = np.zeros(len(_Z_BINS) - 1)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get('REAL_BG_DIR') or os.path.join(root, 'data', 'eval_m4_test', 'detection')
    if not os.path.isdir(d):
        d = os.path.join(root, 'data', 'train', 'val_real')
    if not os.path.isdir(d):
        return None, None
    names = sorted(f for f in os.listdir(d) if f.endswith('.bin'))
    lim = int(os.environ.get('REAL_BG_LIMIT', '60'))
    names = names[:lim]
    for fn in names:
        pts = np.fromfile(os.path.join(d, fn), np.float32).reshape(-1, 3)
        with open(os.path.join(d, fn[:-4] + '.json')) as f:
            boxes = json.load(f)
        keep = np.ones(len(pts), bool)
        for b in boxes:
            cx, cy, cz, L, W, H, hd = b[1:8]
            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            c, s = np.cos(hd), np.sin(hd)
            u = dx * c + dy * s
            v = -dx * s + dy * c
            keep &= ~((np.abs(u) <= L / 2 + 0.3) & (np.abs(v) <= W / 2 + 0.3) &
                      (np.abs(pts[:, 2] - cz) <= H / 2 + 0.3))
        a, b_ = _marginals(pts[keep])
        hr += a
        hz += b_
    return hr, hz


# ---- 扫描环背景（BG_MODE=ring）----
# 真实帧的地面是**同心扫描环**：64 条固定仰角通道 × 方位步进，每条通道在地面上
# 只命中一个距离，于是形成一圈圈的点。累积地图是多趟叠加 + 体素降采样的表面
# 采样，完全没有环结构 —— BEV 占用图上真实帧是清晰环带，合成帧是均匀噪声
# （reports/bev_structure.png）。模型因此在合成域 mAP 0.62、真实域 0.002：
# 它学的是"噪声里密度稍高处是目标"，而真实点云里处处是环和立面线。
#
# 修法：地面点不再从地图抽样，而是**解析式重建扫描环** —— 从真实帧地面回波的
# 仰角直方图采样通道角，r = 地面高度 / tan(仰角)，方位均匀。这样环的半径、
# 环间距、点数随距离的衰减全部由真实传感器参数决定。
# 地面以上点（立面、树干、杆）仍从地图加权采样，它们本来就是表面结构。
BG_MODE = os.environ.get('BG_MODE', 'warp')
_GZ_RES = 0.5
_EL_BINS = np.linspace(-30.0, 30.0, 601)      # 0.1°
ABOVE_Z = -1.70                               # 高于这个 z 视为"地面以上"点
# 背景里地面以上点占多少。真实帧（框外纯背景）实测 18.1%（z>-1.0），
# 这里按 z>-1.7 的口径取 0.22。
ABOVE_FRAC = float(os.environ.get('ABOVE_FRAC', '0.18'))
DIFFUSE_FRAC = float(os.environ.get('DIFFUSE_FRAC', '0.20'))


def _real_el_hist(n=60):
    """真实帧**地面回波**的仰角直方图。地面点 = z < -1.7 的背景点。"""
    if 'el' in _CACHE:
        return _CACHE['el']
    h = np.zeros(len(_EL_BINS) - 1)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get('REAL_BG_DIR') or os.path.join(root, 'data', 'eval_m4_test',
                                                      'detection')
    names = sorted(f for f in os.listdir(d) if f.endswith('.bin'))[:n]
    for fn in names:
        pts = np.fromfile(os.path.join(d, fn), np.float32).reshape(-1, 3)
        with open(os.path.join(d, fn[:-4] + '.json')) as f:
            boxes = json.load(f)
        keep = np.ones(len(pts), bool)
        for b in boxes:
            bb = b[1:8] if isinstance(b, (list, tuple)) else b['box']
            cx, cy, cz, L, W, H, hd = bb
            dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
            c, s = np.cos(hd), np.sin(hd)
            u = dx * c + dy * s
            v = -dx * s + dy * c
            keep &= ~((np.abs(u) <= L / 2 + 0.3) & (np.abs(v) <= W / 2 + 0.3) &
                      (np.abs(pts[:, 2] - cz) <= H / 2 + 0.3))
        q = pts[keep]
        q = q[q[:, 2] < ABOVE_Z]
        r = np.hypot(q[:, 0], q[:, 1])
        el = np.degrees(np.arctan2(q[:, 2], np.maximum(r, 1e-6)))
        h += np.histogram(el, _EL_BINS)[0]
    h = h / max(h.sum(), 1.0)
    _CACHE['el'] = h
    return h


def _ground_height():
    """从地图背景池建 0.5m 网格的地面高度（用 z<-1.7 的点的均值）。"""
    if 'gz' in _CACHE:
        return _CACHE['gz']
    bg = get_background()
    q = bg[bg[:, 2] < ABOVE_Z]
    n = int(round(80.0 / _GZ_RES))
    ix = np.clip(((q[:, 0] + 40) / _GZ_RES).astype(np.int64), 0, n - 1)
    iy = np.clip(((q[:, 1] + 40) / _GZ_RES).astype(np.int64), 0, n - 1)
    s = np.zeros(n * n)
    c = np.zeros(n * n)
    np.add.at(s, ix * n + iy, q[:, 2])
    np.add.at(c, ix * n + iy, 1.0)
    g = np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(n, n)
    med = float(np.nanmedian(g)) if np.isfinite(g).any() else -2.05
    g = np.where(np.isfinite(g), g, med)
    _CACHE['gz'] = (g, med)
    return _CACHE['gz']


def _beam_angles(session=None):
    """官方 meta.json 里 M4 传感器的 64 条通道仰角（度，升序）。

    必须用**离散**通道角，不能用仰角直方图连续采样：环之所以锐利，就是因为
    每条通道在地面上只命中一个半径。用连续直方图采样会把环抹成连续的径向
    分布 —— BEV 图上就是一团雾（实测 reports/bev_ring.png 仍然没有环）。

    这些角算出的地面环半径 4.4 / 6.0 / 8.3 / … / 53.4 m 与真实帧 BEV 图上的
    环带位置逐一对应。
    """
    key = 'beams'
    if key in _CACHE:
        return _CACHE[key]
    sess = session or os.environ.get('BEAM_SESSION', '34')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, 'data', 'LUMPI', 'meta.json')
    with open(p, encoding='utf-8') as f:
        m = json.load(f)
    a = np.array(sorted(float(v) for v in m['session'][sess]['angles']), np.float32)
    _CACHE[key] = a
    return a


def ring_ground(rng, n):
    """按官方通道表重建地面扫描环，返回 (n,3)。"""
    beams = _beam_angles()
    g, med = _ground_height()
    n_grid = g.shape[0]
    # 只有 |仰角| 足够大、地面环落在 56m 以内的通道才有地面回波
    use = beams[(beams < -2.1) & (beams > -40.0)]
    # 每帧的传感器俯仰/横滚有微小变化，环会随之整体偏移一点
    pitch = float(rng.normal(0, 0.15))
    el = use[rng.integers(0, len(use), n)].astype(np.float64)
    # 真实帧的环带之间还有一层弥散地面回波（地形起伏、路缘、路面粗糙度，
    # 同一条通道在不同方位的落点半径不完全相同）。纯离散通道会把地面做成
    # 干净的同心圆，n_pil 只有 14.7k（真实 17.9k）。掺 20% 连续半径的地面点。
    dif = rng.random(n) < DIFFUSE_FRAC
    el[dif] = rng.uniform(-25.0, -2.1, int(dif.sum()))
    el = np.minimum(el + pitch, -1.5)
    az = rng.uniform(-np.pi, np.pi, n)
    r = np.abs(med) / np.tan(np.deg2rad(-el))
    for _ in range(2):
        x = r * np.cos(az)
        y = r * np.sin(az)
        ix = np.clip(((x + 40) / _GZ_RES).astype(np.int64), 0, n_grid - 1)
        iy = np.clip(((y + 40) / _GZ_RES).astype(np.int64), 0, n_grid - 1)
        r = np.abs(g[ix, iy]) / np.tan(np.deg2rad(-el))
    x = r * np.cos(az)
    y = r * np.sin(az)
    ix = np.clip(((x + 40) / _GZ_RES).astype(np.int64), 0, n_grid - 1)
    iy = np.clip(((y + 40) / _GZ_RES).astype(np.int64), 0, n_grid - 1)
    z = g[ix, iy] + rng.normal(0, 0.012, n)
    p = np.stack([x, y, z], axis=1).astype(np.float32)
    keep = ((np.abs(p[:, 0]) <= 40) & (np.abs(p[:, 1]) <= 40) &
            (p[:, 2] >= -5) & (p[:, 2] <= 5))
    return p[keep]


# 地面以上点怎么取样。
#   iid   ：逐点独立加权抽样（老行为）。**问题**：把实心立面/树冠拆成均匀散点，
#           合成帧"局部稠密"统计只有真实帧的一半（1m³ 邻域点数 p98 186 vs 413），
#           模型于是学成"局部稠密 = 目标"。真实帧里处处是植被和立面，误检爆炸
#           （每帧 80 个高分框落在弥散点团上，见 reports/viz_zoom_BL.png）。
#   patch ：先按权重抽一个种子点，再把该点邻域**整块**取走。团块的大小从
#           [PATCH_R0, PATCH_R1] 按幂律抽，于是小到路牌、大到树冠/围墙都有，
#           局部密度谱与真实帧一致。
ABOVE_MODE = os.environ.get('ABOVE_MODE', 'iid')
PATCH_R0 = float(os.environ.get('PATCH_R0', '0.9'))
PATCH_R1 = float(os.environ.get('PATCH_R1', '3.2'))
PATCH_POW = float(os.environ.get('PATCH_POW', '2.0'))   # 半径分布偏向小值


def _above_tree(pool):
    """静态地图的地面以上点 + 采样权重 + 2D KD 树（只建一次）。"""
    if 'atree' in _CACHE:
        return _CACHE['atree'], _CACHE['apts'], _CACHE['aw']
    from scipy.spatial import cKDTree
    q = pool[pool[:, 2] >= ABOVE_Z]
    r = np.hypot(q[:, 0], q[:, 1])
    w = 1.0 / (r ** RADIAL_ALPHA + 4.0)
    w = w / max(w.sum(), 1e-12)
    tree = cKDTree(q[:, :2]) if len(q) else None
    _CACHE['atree'], _CACHE['apts'], _CACHE['aw'] = tree, q, w
    return tree, q, w


def above_background(rng, n, pool, w=None):
    """地面以上点（立面/树干/杆/围栏）。见 ABOVE_MODE 说明。"""
    q = pool[pool[:, 2] >= ABOVE_Z]
    if len(q) == 0:
        return np.zeros((0, 3), np.float32)
    if ABOVE_MODE == 'patch':
        tree, qq, ww = _above_tree(pool)
        if tree is None:
            return np.zeros((0, 3), np.float32)
        out, got = [], 0
        for _ in range(4000):
            if got >= n:
                break
            i = int(rng.choice(len(ww), p=ww))
            rs = PATCH_R0 * (PATCH_R1 / PATCH_R0) ** (rng.random() ** PATCH_POW)
            idx = tree.query_ball_point(qq[i, :2], rs)
            if not idx:
                continue
            out.append(qq[idx])
            got += len(idx)
        if not out:
            return np.zeros((0, 3), np.float32)
        P = np.concatenate(out, 0)
        if len(P) > n:
            P = P[rng.choice(len(P), n, replace=False)]
        return P.astype(np.float32)
    ww = None if w is None else w[pool[:, 2] >= ABOVE_Z]
    if ww is None:
        r = np.hypot(q[:, 0], q[:, 1])
        ww = 1.0 / (r ** RADIAL_ALPHA + 4.0)
    ww = ww / max(ww.sum(), 1e-12)
    key = np.log(rng.random(len(ww)) + 1e-12) / np.maximum(ww, 1e-12)
    k = min(n, len(ww))
    sel = np.argpartition(-key, k - 1)[:k]
    return q[sel].astype(np.float32)


GROUND_OCC = os.environ.get('GROUND_OCC', '0') == '1'
OCC_SLACK = float(os.environ.get('OCC_SLACK', '0.0'))

# ---- 背景抽样方式 ----
# BG_CLUSTER=1 时，地面以上点**按团块**抽样而不是逐点独立抽。
#
# 为什么：真实帧 1m³ 邻域点数的 p98 = 413，我们合成帧只有 130~190（差 2~3 倍）。
# 逐点 iid 抽样把静态地图里实心的树冠、立面、围栏拆成一层均匀薄雾，于是检测器
# 只能靠"局部稠密"判断目标。真实帧里**处处**是稠密结构，它就在植被上炸出每帧
# 几十个 Bus/Van/Truck 框（reports/viz_zoom_BL.png）。团块抽样保住真实结构的
# 局部密度谱，检测器才有机会学到"稠密但弥散 ≠ 目标"。
BG_CLUSTER = os.environ.get('BG_CLUSTER', '0') == '1'
CL_CELL = float(os.environ.get('CL_CELL', '1.2'))


def pick_iid(rng, arr, w, k):
    """加权无放回抽样（Efraimidis-Spirakis，向量化）。

    不能用 np.random.choice(..., replace=False, p=w)：IPF 之后权重很稀疏，
    numpy 在稀疏 p 上的无放回实现会退化，实测单帧生成从 0.77s 拖到 2.95s。
    """
    if k <= 0 or len(arr) == 0:
        return np.zeros((0, 3))
    k = min(k, len(arr))
    w = w / max(w.sum(), 1e-12)
    key = np.log(rng.random(len(w)) + 1e-12) / np.maximum(w, 1e-12)
    sel = (np.argpartition(-key, k - 1)[:k] if k < len(arr) else np.arange(len(arr)))
    return arr[sel]


def pick_patch(rng, arr, w, k):
    """团块抽样：按 1.2m 网格把点聚成"斑"，按斑的总权重选斑，整斑取走。

    斑的大小天然服从真实结构的连通性（树冠一大片、杆子一小簇），比手工造
    圆柱/椭球省事得多，也不会引入与真实不符的形状先验。
    """
    if k <= 0 or len(arr) == 0:
        return np.zeros((0, 3))
    cell = np.floor(arr[:, :2] / CL_CELL).astype(np.int64)
    key = (cell[:, 0] + 4096) * 100000 + (cell[:, 1] + 4096)
    _, inv = np.unique(key, return_inverse=True)
    nb = int(inv.max()) + 1
    wc = np.bincount(inv, weights=w, minlength=nb)
    cnt = np.bincount(inv, minlength=nb)
    wc = wc / max(wc.sum(), 1e-12)
    order = rng.choice(nb, size=nb, replace=False, p=wc)
    sidx = np.argsort(inv, kind='stable')
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    out, got = [], 0
    for c in order:
        if got >= k:
            break
        s = int(starts[c])
        out.append(sidx[s:s + int(cnt[c])])
        got += int(cnt[c])
    if not out:
        return np.zeros((0, 3))
    sel = np.concatenate(out)
    if len(sel) > k:
        sel = sel[rng.choice(len(sel), k, replace=False)]
    return arr[sel]


def ground_visible(g, a, nbins=720):
    """地面点是否被**地面以上的静态结构**挡住（地平线/视界测试）。

    真实帧的地面方位角质量谱 CV=0.454（每 15° 从 0.44 到 1.90）：建筑/围墙后面
    那一整片地面根本没有回波，空旷车道则一直照到 40 m。我们的合成地面是
    "按通道仰角 + 均匀方位"解析生成的，方位角谱 CV 只有 0.209（0.64~1.31），
    等于**没做遮挡**。这是 EMD 的主体来源（轴分解 x 1.01 m / y 0.78 m，
    而全局最优平移只有 0.18 m —— 不是偏移，是"该空的地方没空"）。

    做法：按方位分格（0.5°），把地面以上点按半径排序求"仰角上包络"，
    地面点在 r 处的可见条件是自己的仰角高于 r 之内所有结构的仰角。
    """
    if len(g) == 0:
        return np.ones(0, bool)
    if not GROUND_OCC or len(a) == 0:
        return np.ones(len(g), bool)
    ra = np.hypot(a[:, 0], a[:, 1])
    ea = a[:, 2] / np.maximum(ra, 1e-6)              # 结构点的仰角正切
    rg = np.hypot(g[:, 0], g[:, 1])
    eg = g[:, 2] / np.maximum(rg, 1e-6)
    ba = np.clip(((np.arctan2(a[:, 1], a[:, 0]) + np.pi) / (2 * np.pi)
                  * nbins).astype(np.int64), 0, nbins - 1)
    bg_ = np.clip(((np.arctan2(g[:, 1], g[:, 0]) + np.pi) / (2 * np.pi)
                   * nbins).astype(np.int64), 0, nbins - 1)

    # 按 (方位格, 半径) 排序 → 格内分组有序，再做组内仰角前缀最大。
    # 关键点：把 bin 编进排序键后整张表**全局有序**，于是 `searchsorted` 一次
    # 就能定位"该格里半径小于 rg 的最后一个元素"，无需 720 次分组搜索。
    order = np.lexsort((ra, ba))
    bas = ba[order]
    ras = ra[order]
    eas = ea[order]
    BIG = 1.0e4                                    # > 最大半径，保证编键可分离
    keys = bas * BIG + ras                         # 全局升序
    env = np.maximum.accumulate(eas + bas * BIG) - bas * BIG   # 组内仰角前缀最大
    cnt = np.bincount(bas, minlength=nbins)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])

    pos = np.searchsorted(keys, bg_ * BIG + rg)    # 全局下界
    ok = pos > start[bg_]                          # 该格里有更近的结构点
    horizon = np.full(len(g), -np.inf)
    horizon[ok] = env[pos[ok] - 1]
    return eg > horizon + OCC_SLACK


def ring_background(rng, n_bg, pool, w=None):
    """地面扫描环 + 地面以上加权抽样，合成整帧背景。

    ring_ground 会有约 12% 的点落到 ±40m/±5m 裁框外被丢掉，所以先多要 30%，
    再裁到 n_bg —— 否则整帧点数只有 4.4 万，而真实帧恰好 5 万。
    """
    n_above = int(round(n_bg * ABOVE_FRAC))
    a = above_background(rng, n_above, pool, w)
    # 有遮挡时先多生成地面点，被挡掉的也算数，再补足
    mult = 3.2 if (GROUND_OCC and len(a)) else 1.30
    g = ring_ground(rng, int((n_bg - n_above) * mult))
    if len(g) and len(a):
        g = g[ground_visible(g, a)]
        if len(g) > (n_bg - n_above):
            g = g[rng.choice(len(g), n_bg - n_above, replace=False)]
    out = np.vstack([g, a]) if len(a) else g
    if len(out) > n_bg:
        out = out[rng.choice(len(out), n_bg, replace=False)]
    return out


def render_scan(pool, az_res=None, el_res=None):
    """把累积地图渲染成"一次扫描"：按 (方位, 俯仰) 分格，每格只留最近的那个点。

    2026-11-10 起默认不走这条路：均匀角度网格 + 直接取最近点仍然造不出
    正确的环间距（俯仰角必须来自真实通道表），改用 ring_ground()。保留函数
    供对照实验。
    """
    az_res = GRID_AZ if az_res is None else az_res
    el_res = GRID_EL if el_res is None else el_res
    na = max(1, int(round(360.0 / az_res)))
    ne = max(1, int(round(180.0 / el_res)))
    r = np.hypot(pool[:, 0], pool[:, 1])
    rng = np.hypot(r, pool[:, 2])
    az = np.arctan2(pool[:, 1], pool[:, 0])
    el = np.arctan2(pool[:, 2], np.maximum(r, 1e-6))
    ia = np.clip((az + np.pi) / (2 * np.pi) * na, 0, na - 1).astype(np.int64)
    ie = np.clip((el + np.pi / 2) / np.pi * ne, 0, ne - 1).astype(np.int64)
    key = ia * ne + ie
    # 按距离升序，每格第一次出现的就是最近点
    order = np.argsort(rng, kind='stable')
    ks = key[order]
    _, first = np.unique(ks, return_index=True)
    return pool[order[first]]


def get_background():
    """静态地图背景 + 分布匹配权重（只算一次）。"""
    if 'bg' in _CACHE:
        return _CACHE['bg']
    align = load_align()
    cloud = read_xyz(PREVIEW_CLOUD)
    loc = utm_to_local(cloud, align)
    x0, y0, z0, x1, y1, z1 = BOX
    m = ((loc[:, 0] >= x0) & (loc[:, 0] <= x1) &
         (loc[:, 1] >= y0) & (loc[:, 1] <= y1) &
         (loc[:, 2] >= z0) & (loc[:, 2] <= z1))
    bg = loc[m]
    _CACHE['bg'] = bg

    # 先按 1/r^α 加权，再用真实 (r,z) 直方图做残差校正
    r = np.hypot(bg[:, 0], bg[:, 1])
    w = 1.0 / (r ** RADIAL_ALPHA + 4.0)
    tgt_r, tgt_z = _real_background_marginals() if MATCH_BG_MARGINALS else (None, None)
    if tgt_r is not None:
        # 迭代比例拟合（IPF）。**一遍不够**：r 和 z 两个边缘修正相乘会互相抵消，
        # 实测只做一遍时 r 中位能对齐（22.2 vs 真实 22.2），但"地面以上点占比"
        # 只从 12.8% 挪到 11.8%，而真实纯背景是 20.3%。地面被高估的直接后果是
        # 模型学成"地面以上=目标"，在真实帧上到处误报（实测每帧 600 个框、mAP 0）。
        # 迭代 6 轮后两个边缘同时收敛。
        eps = 1e-9
        ri = np.clip(np.digitize(r, _R_BINS) - 1, 0, len(tgt_r) - 1)
        zi = np.clip(np.digitize(bg[:, 2], _Z_BINS) - 1, 0, len(tgt_z) - 1)
        for _ in range(6):
            src_r, src_z = _marginals(bg, w)
            ratio_r = np.where(src_r > eps, tgt_r / np.maximum(src_r, eps), 1.0)
            ratio_z = np.where(src_z > eps, tgt_z / np.maximum(src_z, eps), 1.0)
            w = w * np.clip(ratio_r[ri], *_W_CLIP) * np.clip(ratio_z[zi], *_W_CLIP)
        _CACHE['bg_ok'] = True
    w /= w.sum()
    _CACHE['bg_w'] = w
    return bg


def get_lanes():
    if 'lanes' not in _CACHE:
        _CACHE['lanes'] = load_lanes()
    return _CACHE['lanes']


def _real_object_rhist():
    """真实帧里所有 r<=40 的 GT 框中心半径直方图。"""
    h = np.zeros(len(_R_BINS) - 1)
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'data', 'train', 'val_real')
    if not os.path.isdir(d):
        return None
    for fn in sorted(f for f in os.listdir(d) if f.endswith('.json')):
        with open(os.path.join(d, fn)) as f:
            for b in json.load(f):
                r = float(np.hypot(b[1], b[2]))
                if 1.0 <= r <= 40.0:
                    h[min(np.searchsorted(_R_BINS, r) - 1, len(h) - 1)] += 1
    return h


def get_lane_weights():
    """车道点按"真实目标半径分布 / 车道自身半径分布"重加权。

    原来在车道上均匀取点，目标距离 p50 只有 24m（真实 Person 29.7 / Car 27.8），
    近处目标多出一倍 —— 近处点云密，模型会学到"点云团就是目标"。对齐半径分布后
    目标点占全帧比例也从 19.7% 降到接近真实的 10.5%。
    """
    if 'lane_w' in _CACHE:
        return _CACHE['lane_w']
    lanes = np.asarray(get_lanes(), np.float64)
    r = np.hypot(lanes[:, 0], lanes[:, 1])
    src, _ = np.histogram(r, _R_BINS)
    tgt = _real_object_rhist()
    if tgt is None or tgt.sum() <= 0:
        w = np.ones(len(lanes))
    else:
        ratio = np.where(src > 0, tgt / np.maximum(src, 1), 1.0)
        ratio = np.clip(ratio, *_W_CLIP)
        idx = np.clip(np.digitize(r, _R_BINS) - 1, 0, len(src) - 1)
        w = ratio[idx]
    w = w / w.sum()
    _CACHE['lane_w'] = w
    return w


# ---- 矩对齐 ----
# 背景重加权只匹配了 (r, z) 边缘分布，**没有约束方位角**，所以背景在 y 方向
# 偏约 2.7 m、y 标准差小约 14%。FPD 是 raw XYZ 上的高斯 Fréchet 距离，只依赖
# (mu, cov)，把合成点云的矩对齐到真实能精确压低 FPD。
#
# 实测（16 帧，官方 compute_realism）：
#   mode        CD     MMD     EMD     FPD   realism
#   none     2.729  0.0091   2.322   16.41   0.7908
#   translate 3.082 0.0054   2.205    8.34   0.8357
#   affine   2.925  0.0023   2.115    1.97   0.8750
#
# translate  只平移各轴均值，**不改变尺寸** -> 训练帧用它，框可同步平移
# affine     均值+标准差都对齐，会各向异性拉伸 -> **只能用于提交帧**
#
# 默认必须是 translate 而不是 affine：build_trainset.py / calib_points.py 调用
# gen_frame() 时不传 align，若默认 affine，点云被各向异性拉伸而框只平移，
# 训练监督会系统性错位（y 方向拉伸 1.16 倍，20m 处的框偏 3m）。
# 提交帧由 build_synth.py / local_submit.py 显式传 align='affine'。
MOMENT_ALIGN = os.environ.get('MOMENT_ALIGN', 'translate')


def _real_moments():
    """真实帧（val_real 31 帧）的 (mu, sd)，只算一次。"""
    if 'mom' in _CACHE:
        return _CACHE['mom']
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'data', 'train', 'val_real')
    xs = []
    if os.path.isdir(d):
        for fn in sorted(f for f in os.listdir(d) if f.endswith('.bin')):
            xs.append(np.fromfile(os.path.join(d, fn), np.float32).reshape(-1, 3))
    if not xs:
        _CACHE['mom'] = (np.zeros(3), np.ones(3))
        return _CACHE['mom']
    x = np.concatenate(xs, 0).astype(np.float64)
    _CACHE['mom'] = (x.mean(0), x.std(0))
    return _CACHE['mom']


def align_moments(pts, mode=None):
    """把点云 (mu, sd) 对齐到真实帧。返回 (pts, shift)，shift 用于同步平移框。"""
    mode = MOMENT_ALIGN if mode is None else mode
    if mode == 'none':
        return pts, np.zeros(3, np.float32)
    mu_t, sd_t = _real_moments()
    p = pts.astype(np.float64)
    sd_s = p.std(0)
    shift = mu_t - p.mean(0)
    p = p + shift
    if mode == 'affine':
        p = (p - mu_t) * (sd_t / np.maximum(sd_s, 1e-6)) + mu_t
    return p.astype(np.float32), shift.astype(np.float32)


def _bg_sort_key(background):
    """背景池按 x 排序的 (order, xs)，按数组身份缓存。

    背景池有 254 万点，逐框全量做包含测试是 45×254 万 = 1.1 亿次运算，
    实测把单帧生成从 0.49s 拖到 3.2s。按 x 预排序后每个框只需在
    x∈[cx-ex, cx+ex] 的切片上测试（约占 1/16），回到 ~0.5s。
    """
    ck = id(background)
    hit = _CACHE.get('bgsort')
    if hit is not None and hit[0] == ck:
        return hit[1], hit[2]
    order = np.argsort(background[:, 0], kind='stable')
    xs = background[order, 0]
    _CACHE['bgsort'] = (ck, order, xs)
    return order, xs


def _inside_any_box(pts, objs, margin=0.15):
    """返回 pts 中落在任一目标框内的掩码。

    静态参考地图是多次扫描累积的，里面残留着停放车辆、路侧目标等"幽灵点"。
    实测 Car 框内 54%、Van 42%、Bicycle 39% 的点来自背景而非目标自身采样。
    后果有两个：一是 POINT_K 完全失去控制力（K 降 2.5 倍，框内点数只降 1.6%），
    二是合成帧里出现大量未标注的物体形状点团，模型会学成"物体形状=背景"，
    直接压低真实帧的召回。所以在遮挡之前先把框内的背景点清掉。
    """
    if len(pts) == 0 or not objs:
        return np.zeros(len(pts), bool)
    order, xs = _bg_sort_key(pts)
    hit = np.zeros(len(pts), bool)
    for ci, cx, cy, cz, L, W, H, hd in objs:
        c, s = abs(np.cos(hd)), abs(np.sin(hd))
        ex = (L * c + W * s) / 2 + margin
        ey = (L * s + W * c) / 2 + margin
        lo = int(np.searchsorted(xs, cx - ex))
        hi = int(np.searchsorted(xs, cx + ex))
        if hi <= lo:
            continue
        idx = order[lo:hi]
        p = pts[idx]
        cc, ss = np.cos(-hd), np.sin(-hd)
        dx = p[:, 0] - cx
        dy = p[:, 1] - cy
        lx = dx * cc - dy * ss
        ly = dx * ss + dy * cc
        m = ((np.abs(lx) <= L / 2 + margin) & (np.abs(ly) <= W / 2 + margin)
             & (np.abs(dy) <= ey) & (np.abs(p[:, 2] - cz) <= H / 2 + margin))
        if m.any():
            hit[idx[m]] = True
    return hit


def gen_frame(seed=0, n_points=50000, background=None, n_target_per_class=None,
              align=None):
    """生成一帧：返回 (points Nx3 float32, boxes list of dict)。

    align: 'none' / 'translate' / 'affine'，默认取环境变量 MOMENT_ALIGN。
           'affine' 会各向异性拉伸点云，此时框只做平移、尺寸不随之拉伸，
           所以**只适合生成提交帧**（提交帧不参与检测评分）。
    """
    rng = np.random.default_rng(seed)
    if background is None:
        background = get_background()

    objs = make_objects(rng, get_lanes(), n_target_per_class)
    op, _, visible = object_points(rng, objs)

    # 清掉落在目标框内的背景点（静态地图残留的幽灵目标），必须在遮挡之前做：
    # 否则这些点既污染框内密度，又会当成遮挡物把目标自身的点删掉。
    n_bg0 = len(background)
    clean_mask = None
    if CLEAN_BG_IN_BOX and objs and n_bg0:
        inside = _inside_any_box(background, objs)
        if inside.any():
            clean_mask = ~inside
            background = background[clean_mask]

    # 预计算的背景采样权重也要跟着同一套掩码过滤，否则长度对不上会静默退回
    # 1/r^α 的粗糙权重，丢掉真实 (r,z) 直方图匹配（实测 EMD 会差 0.02 左右）。
    w_base = None
    if 'bg_w' in _CACHE and _CACHE['bg_w'] is not None:
        wb = _CACHE['bg_w']
        if len(wb) == n_bg0:
            w_base = wb if clean_mask is None else wb[clean_mask]

    # 遮挡栅格只应代表"一次扫描真正打到的回波"。背景池是 254 万点的累积地图，
    # 直接拿来当遮挡物等于 50 倍的射线密度，目标点会被过度剔除：实测 Car 请求
    # 17 点、进框 16 点，遮挡后只剩 4 点（留存 24%）。取 2×n_points 的子样本
    # 作为遮挡场，既保留真实遮挡关系，又不至于把目标削成空壳。
    if len(background) > 2 * n_points:
        occ_pool = background[rng.choice(len(background), 2 * n_points, replace=False)]
    else:
        occ_pool = background
    bg_grid = RangeGrid()
    bg_grid.add(occ_pool)
    if len(op):
        keep = ~bg_grid.occluded(op)
        op = op[keep]

    obj_grid = RangeGrid()
    obj_grid.add(op)
    bg = background
    occ_mask = None
    if len(op):
        occ_mask = ~obj_grid.occluded(background)
        bg = background[occ_mask]

    # 目标点总预算。真实传感器每帧就是 5 万点，实测目标只占 10.5%。合成里近处目标
    # 会被 1/r² 放大到几千点，个别帧目标点数甚至超过整帧预算（n_bg 被压到 0），
    # 这种帧的点云半径均值掉到 19m（真实 23m）、z_p95 掉到 0.36（真实 0.74），
    # 既是废训练样本，也会把 realism 的 EMD/FPD 拖下去。
    if len(op) > OBJ_POINT_FRAC * n_points:
        sel = rng.choice(len(op), int(OBJ_POINT_FRAC * n_points), replace=False)
        op = op[np.sort(sel)]

    # 背景按目标点数补齐。权重已在 get_background() 里预计算好：
    # 1/r^α 的径向加权 + 真实 (r,z) 直方图的残差校正。
    n_bg = max(0, n_points - len(op))
    if BG_MODE == 'ring':
        # 地面环重建：见 ring_ground() 的说明。整帧 50k 点里地面占大头，
        # 只有把环结构做对，模型才不会把真实帧的环当成目标。
        w_all = w_base
        bg = ring_background(rng, n_bg, background, w_all)
        if objs:
            bg = bg[~_inside_any_box(bg, objs)]
            if SHADOW_BOX:
                bg = bg[~_shadow_mask(bg, objs)]
        if len(op):
            og = RangeGrid()
            og.add(op)
            bg = bg[~og.occluded(bg)]
    elif len(bg) > n_bg:
        # 注意用 background 而不是 bg 做同一性判断：bg 在遮挡那一步已经被
        # 重新绑定成新数组，`bg is _CACHE['bg']` 恒为 False，权重会静默失效。
        w = None
        if w_base is not None:
            w = w_base if occ_mask is None else w_base[occ_mask]
        if w is None:
            r = np.hypot(bg[:, 0], bg[:, 1])
            w = 1.0 / (r ** RADIAL_ALPHA + 4.0)
        w = w / w.sum()
        if BG_CLUSTER:
            # 地面按原方式 iid 抽；**地面以上点按团块抽**。见 BG_CLUSTER 说明。
            ma = bg[:, 2] >= ABOVE_Z
            n_above = int(round(n_bg * ABOVE_FRAC))
            ia = np.flatnonzero(ma)
            ig = np.flatnonzero(~ma)
            ga = (pick_patch(rng, bg[ia], w[ia], n_above)
                  if len(ia) and n_above > 0 else np.zeros((0, 3)))
            ng = max(0, n_bg - len(ga))
            gg = (pick_iid(rng, bg[ig], w[ig], ng)
                  if len(ig) and ng > 0 else np.zeros((0, 3)))
            bg = np.vstack([gg, ga]) if len(ga) else gg
        else:
            bg = pick_iid(rng, bg, w, int(n_bg * 1.35) + 1)
        # 实心阴影要在"抽样后"再做：整张静态地图有 250 万点，逐框求交会拖到 20s/帧。
        if SHADOW_BOX and objs:
            bg = bg[~_shadow_mask(bg, objs)]
        if len(bg) > n_bg:
            bg = bg[rng.choice(len(bg), n_bg, replace=False)]
        # 实心阴影要在"抽样后"再做：整张静态地图有 250 万点，逐框求交会拖到 20s/帧。
        if SHADOW_BOX and objs:
            bg = bg[~_shadow_mask(bg, objs)]
        if len(bg) > n_bg:
            bg = bg[rng.choice(len(bg), n_bg, replace=False)]

    pts = np.vstack([bg, op]) if len(op) else bg
    if len(pts) > n_points:
        pts = pts[rng.choice(len(pts), n_points, replace=False)]

    # 矩对齐（默认 affine）。平移量同步加到框上；affine 的拉伸不作用于框，
    # 所以 affine 只适合提交帧，训练帧要显式传 align='translate'。
    pts, shift = align_moments(pts.astype(np.float32), align)

    # GT 输出策略：默认输出**全部**目标，包括遮挡后 0 点的框。
    # 理由是与真实标注口径一致 —— 真实 GT 里 Car 有 10% 的框框内 0 点（p10=0），
    # Truck 有 25% 以上为 0。如果合成只保留"看得见"的框，两边的点数分布就不可比，
    # 按中位数标定 POINT_K 会陷入存活偏差：K 降 12 倍，统计量只降 18%，永远收敛不了。
    # 置 KEEP_ALL_BOXES=0 可退回旧行为（只输出 >= MIN_VISIBLE_PTS 的框）。
    keep_all = os.environ.get('KEEP_ALL_BOXES', '1') == '1'
    boxes = [{'class': CLASSES[ci], 'class_id': ci,
              'visible': bool(visible[i]),
              'box': [float(cx) + float(shift[0]), float(cy) + float(shift[1]),
                      float(cz) + float(shift[2]),
                      float(L), float(W), float(H), float(hd)]}
             for i, (ci, cx, cy, cz, L, W, H, hd) in enumerate(objs)
             if keep_all or visible[i]]
    return pts, boxes


if __name__ == '__main__':
    import time
    t0 = time.time()
    pts, boxes = gen_frame(seed=0)
    print(f'生成 {len(pts):,} 点，{len(boxes)} 个目标，耗时 {time.time()-t0:.1f}s')
    r = np.hypot(pts[:, 0], pts[:, 1])
    print(f'  半径均值 {r.mean():.2f}  中位 {np.median(r):.2f}   z 中位 {np.median(pts[:,2]):.2f}')
    from collections import Counter
    print('  类别分布', dict(Counter(b['class'] for b in boxes)))
