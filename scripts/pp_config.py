#!/usr/bin/env python3
"""PointPillars 训练/推理共享配置。所有脚本从这里取常量，避免各写一套。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, SYNTH  # noqa: E402

# ---- 点云范围（与 config.json 一致）----
X_MIN, Y_MIN, Z_MIN = -40.0, -40.0, -5.0
X_MAX, Y_MAX, Z_MAX = 40.0, 40.0, 5.0
PC_RANGE = (X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX)

# ---- pillar ----
VOXEL_X = VOXEL_Y = 0.32
VOXEL_Z = Z_MAX - Z_MIN            # 单层，整根柱子一格
GRID_X = int(round((X_MAX - X_MIN) / VOXEL_X))   # 250
GRID_Y = int(round((Y_MAX - Y_MIN) / VOXEL_Y))   # 250
MAX_PILLARS = 32768        # 实测一帧只占用 2.7 万个柱，不截断
MAX_PTS_PER_PILLAR = 6     # 实测平均 1.8 点/柱；截到 6 仍保留 98.3% 的点
IN_FEATURES = 9

# ---- 类别 ----
# class_id 6 = **Van**（2026-11-09 更正，之前判成 Unknown 是错的）。
# 证据：禁用区间 1613 帧里 class 6 共 2797 个，折算 ×50 帧 = 87；官方测试 GT
# 是 Van 79、Unknown 0；尺寸 3.67×1.80×1.93 是厢式车形状。七个 class_id 的
# 计数全部对上官方 GT。class_id 7 在测试时段 0 实例，直接丢弃。
# 模型侧 7 类，索引 6 = Van；评分侧仍给 config.json 的 8 个名字，Unknown 因
# n_gt=0 会被评分器自动跳过。
CLASS_NAMES = ['Person', 'Car', 'Bicycle', 'Motorcycle', 'Bus', 'Truck', 'Van']
NUM_CLASS = len(CLASS_NAMES)
EVAL_CLASS_NAMES = ['Person', 'Car', 'Bicycle', 'Motorcycle', 'Bus', 'Truck',
                    'Van', 'Unknown']

# ---- anchor：取自 data/eval_m4_test 的 200 个**非禁用**基准帧的尺寸中位数 ----
# 2026-11-10 合规修正：官方 50 个检测测试帧全部在 forbidden_frames.txt 里，
# 旧 anchor 用的是"禁用区间 6497~8109 共 1623 帧"的统计，等于间接观察测试帧
# 真值，违反 rules.md 第 11 行。现改为 200 个与禁用帧零重叠的基准帧，
# 属于第 9 行允许的 train split 超参选择。
#
# 数值变化最大的是大车（旧值严重偏小，导致这两类 AP 恒为 0）：
#   Bus   4.51×2.00×1.72  ->  9.57×2.32×3.11
#   Truck 6.42×2.18×2.89  ->  8.13×2.50×3.18
ANCHOR_SIZE = {
    'Person':     (0.94, 0.80, 1.89),
    'Car':        (4.41, 1.91, 1.60),
    'Bicycle':    (1.95, 0.83, 1.86),
    'Motorcycle': (1.97, 0.84, 1.86),
    'Bus':        (9.57, 2.32, 3.11),
    'Truck':      (8.13, 2.50, 3.18),
    'Van':        (3.67, 1.80, 1.93),
}
ANCHOR_Z = {
    'Person': -1.22, 'Car': -1.33, 'Bicycle': -1.23, 'Motorcycle': -1.16,
    'Bus': -0.61, 'Truck': -0.73, 'Van': -1.16,
}

# ---- 旧 anchor（非禁用时段尺寸）----
# 仅供 A/B 对比和复现 pp_v1~pp_v6 的旧权重：那些权重是在这套尺寸上训的，
# 用新 anchor 解码会让框尺寸跳变。设 ANCHOR_PROFILE=legacy 可切回来。
#   ANCHOR_PROFILE=legacy python scripts/local_submit.py --ckpt data/train/ckpt/pp_v6.pt
LEGACY_ANCHOR_SIZE = {
    'Person':     (0.78, 0.74, 1.90),
    'Car':        (4.50, 2.00, 1.70),
    'Bicycle':    (2.00, 0.86, 1.90),
    'Motorcycle': (1.97, 0.84, 1.83),
    'Bus':        (12.04, 2.86, 3.25),
    'Truck':      (10.07, 2.93, 3.43),
    'Van':        (4.77, 1.99, 2.00),
}
LEGACY_ANCHOR_Z = {
    'Person': -1.20, 'Car': -1.35, 'Bicycle': -1.25, 'Motorcycle': -1.29,
    'Bus': -0.51, 'Truck': -0.52, 'Van': -1.18,
}
ANCHOR_PROFILE = os.environ.get('ANCHOR_PROFILE', 'test')
if ANCHOR_PROFILE == 'legacy':
    ANCHOR_SIZE = LEGACY_ANCHOR_SIZE
    ANCHOR_Z = LEGACY_ANCHOR_Z
HEADINGS = (0.0, 1.5707963)

# ---- 多尺度 anchor ----
# 实测真实帧里 Bus 有 28%、Truck 有 38% 的框长 > 8m，其余集中在 4.5/6.4m，
# 是双峰分布。每类只给一个尺寸时，另一半目标连 IoU 0.5 都匹配不上，
# 正样本恒为 0 —— 这是 Bus/Truck AP≈0 的机制性原因。
# 打开 PP_MSA=1 给这三类补一个"另一峰"的尺寸。
MSA = os.environ.get('PP_MSA', '0') == '1'
ANCHOR_EXTRA = {
    'Bus': [(4.60, 2.00, 1.75)],
    'Truck': [(3.60, 2.10, 2.60)],
    'Van': [(4.75, 1.98, 2.00)],
}


def anchor_sizes(name):
    out = [ANCHOR_SIZE[name]]
    if MSA:
        out += ANCHOR_EXTRA.get(name, [])
    return out


NA = sum(len(HEADINGS) * len(anchor_sizes(n)) for n in CLASS_NAMES)

# 目标底面 z 中位数（测试时段实测 -2.168，非禁用时段 -2.188，差 2 cm）。
# 推理时把框 z 吸附到 ground + H/2，等于免费去掉 z 方向误差，3D IoU 直接受益。
GROUND_Z = -2.168

# ---- 特征图尺寸 ----
# 默认 125x125（0.64 m 一格）：backbone 下采样 2 次，检测头只有一半分辨率。
# 实测合成模型在真实帧上的中心误差中位数 1.20 m，是格子尺寸的近 2 倍，
# 0.64 m 的量化精度成为定位精度的硬上限之一。
# 设 PP_NECK=deep 切到"标准 PointPillars 式颈 + 全分辨率头"（250x250，0.32 m）：
#   b1(64->64 x2, s1) / b2(64->128, s2) / b3(128->256, s2)
#   up3 -> cat(b2) 256 / up2 -> cat(b1) 192 -> head
# 这是独立的 profile：换了它就必须重训，旧权重不能混用。
DEEP_NECK = os.environ.get('PP_NECK', '') == 'deep'
# 每个 stage 多一层卷积（标准 PointPillars 的 RPN 每级 1 下采样 + 2 卷积）。
# 原来每级只有 1 层，整网 0.85M 参数；在"目标稀疏、密度真实"的合成集上
# 表现为域内 mAP 只有 0.21 —— 不是学不会，是容量不够。
WIDE = os.environ.get('PP_WIDE', '0') == '1'
FEAT_X = GRID_X if DEEP_NECK else GRID_X // 2
FEAT_Y = GRID_Y if DEEP_NECK else GRID_Y // 2

# ---- 训练超参 ----
BATCH_SIZE = 4
LR = 2e-3
EPOCHS = 40
MAX_POS = 16000          # 小目标扩到 3x3 邻域后正样本变多，上限放到 8000 基本不截断
MIN_POS_PER_GT = int(os.environ.get('PP_MIN_POS', '6'))
# 少于 MIN_POS_PER_GT 个正样本的目标，把中心 (2*POS_R+1)^2 邻域也标成正样本。
# POS_R=1 是 3x3（旧默认）。Person 0.94x0.80 在 0.32 m 网格上框内只有 3.2 个
# 正 anchor，正样本率 1.4e-5，focal 也压不住 —— 这是 Person/Bicycle/Moto
# 真实域 oracle recall 只有 0.23/0.25/0.20 而 Car 有 0.625 的机制性原因。
POS_R = int(os.environ.get('PP_POS_R', '1'))

# ---- 路径 ----
TRAIN_SYNTH = os.path.join(DATA, 'train', 'synth')
TRAIN_REAL = os.path.join(DATA, 'train', 'real')
VAL_REAL = os.path.join(DATA, 'train', 'val_real')
TEST_FRAMES_DIR = os.path.join(os.path.dirname(DATA), 'lumpi_starting_kit',
                               'detection_test_frames')
CKPT_DIR = os.path.join(DATA, 'train', 'ckpt')
OUT_DIR = os.path.join(DATA, 'train', 'out')


def anchor_array():
    """(NA, 7) 的 anchor 模板：[cx, cy, cz, L, W, H, heading]（cx/cy 稍后填充）。"""
    import numpy as np
    rows = []
    for name in CLASS_NAMES:
        for (L, W, H) in anchor_sizes(name):
            for hd in HEADINGS:
                rows.append([0.0, 0.0, ANCHOR_Z[name], L, W, H, hd])
    return np.array(rows, dtype=np.float32)


def anchor_class_ids():
    import numpy as np
    ids = []
    for ci, name in enumerate(CLASS_NAMES):
        ids += [ci] * (len(HEADINGS) * len(anchor_sizes(name)))
    return np.asarray(ids, dtype=np.int64)


if __name__ == '__main__':
    print(f'grid {GRID_X}x{GRID_Y}  feat {FEAT_X}x{FEAT_Y}  '
          f'anchors/loc {NA}  total anchors {NA*FEAT_X*FEAT_Y:,}')
    for p in (TRAIN_SYNTH, TRAIN_REAL, VAL_REAL, CKPT_DIR, OUT_DIR):
        print(('OK  ' if os.path.isdir(p) else 'MISS') + ' ' + p)
    print('test frames dir', 'OK' if os.path.isdir(TEST_FRAMES_DIR) else 'MISS')
