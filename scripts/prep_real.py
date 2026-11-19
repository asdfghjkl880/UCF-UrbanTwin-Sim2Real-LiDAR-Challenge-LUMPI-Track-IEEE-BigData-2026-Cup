#!/usr/bin/env python3
"""
把 data/test_data 的真实 PLY 帧转成与合成集一致的训练格式。

变换与官方测试帧完全一致：裁剪到 ±40m / ±5m，再子采样到恰好 50,000 点。
Measurement4 的 31 帧留作真实域验证集（它与 50 个测试帧同传感器、同路口、
不同时段，且帧号 0~30 与测试帧 6498~8100 完全不重叠），其余 6 个 measurement
作为真实域训练数据。

用法：
    python scripts/prep_real.py
输出：
    data/train/val_real/<fid>.bin + labels.json      （M4）
    data/train/real/<meas>_<fid>.bin + labels.json   （其余 6 个）
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LABEL_DIR, TEST_DATA, MEASUREMENTS  # noqa: E402
from refcloud import parse_ply_header  # noqa: E402
import pp_config as C  # noqa: E402

VAL_MEAS = 'Measurement4'
N_POINTS = 50000


def read_ply_xyz(path):
    n, dt, off = parse_ply_header(path)
    raw = np.fromfile(path, dtype=dt, count=n, offset=off)
    return np.stack([raw['x'], raw['y'], raw['z']], axis=1).astype(np.float32)


def load_labels(meas):
    """{time: [[cls, cx, cy, cz, L, W, H, hd], ...]}"""
    out = {}
    with open(os.path.join(LABEL_DIR, meas, 'Label.csv'),
              encoding='utf-8', errors='replace') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            try:
                tm = round(float(row[0]), 3)
                cls = int(float(row[7]))
                cx, cy, cz = float(row[9]), float(row[10]), float(row[11])
                L, W, H = float(row[12]), float(row[13]), float(row[14])
                hd = float(row[15])
            except (ValueError, IndexError):
                continue
            if not (L > 0.1 and W > 0.1 and H > 0.1):
                continue
            if cls > 6:      # class 7 是极罕见的 Van 形，全库 121 个，模型侧只有 7 类
                continue
            if not np.isfinite([cx, cy, cz, L, W, H, hd]).all():
                continue
            out.setdefault(tm, []).append([cls, cx, cy, cz, L, W, H, hd])
    return out


def crop_subsample(pts, rng, n_out=N_POINTS):
    m = ((pts[:, 0] >= C.X_MIN) & (pts[:, 0] <= C.X_MAX) &
         (pts[:, 1] >= C.Y_MIN) & (pts[:, 1] <= C.Y_MAX) &
         (pts[:, 2] >= C.Z_MIN) & (pts[:, 2] <= C.Z_MAX))
    p = pts[m]
    if len(p) > n_out:
        p = p[rng.choice(len(p), n_out, replace=False)]
    return p


def in_range(b):
    _, cx, cy, cz, L, W, H, _ = b
    return (C.X_MIN + L / 2 <= cx <= C.X_MAX - L / 2 and
            C.Y_MIN + W / 2 <= cy <= C.Y_MAX - W / 2 and
            C.Z_MIN <= cz <= C.Z_MAX)


def main():
    rng = np.random.default_rng(2026)
    for d in (C.VAL_REAL, C.TRAIN_REAL):
        os.makedirs(d, exist_ok=True)

    total = {'val': 0, 'train': 0}
    for meas in MEASUREMENTS:
        lidar = os.path.join(TEST_DATA, meas, 'lidar')
        if not os.path.isdir(lidar):
            print(f'  跳过 {meas}（无 lidar 目录）')
            continue
        labs = load_labels(meas)
        keys = np.array(sorted(labs.keys()))
        is_val = (meas == VAL_MEAS)
        out_dir = C.VAL_REAL if is_val else C.TRAIN_REAL
        n_ok = 0

        for fn in sorted(os.listdir(lidar)):
            if not fn.endswith('.ply'):
                continue
            fid = int(os.path.splitext(fn)[0])
            tm = round(fid / 10.0, 3)
            if len(keys) == 0:
                continue
            k = keys[int(np.argmin(np.abs(keys - tm)))]
            if abs(k - tm) > 0.15:
                continue
            pts = read_ply_xyz(os.path.join(lidar, fn))
            pts = crop_subsample(pts, rng)
            if len(pts) < 1000:
                continue
            # 保留全部框，不做范围过滤——评分器的 GT 就是全量框（含 ±40m 之外、
            # 裁剪后看不到的目标）。训练时再在 Dataset 里按点云范围过滤，
            # 验证时直接用全量框，这样本地 mAP 和真实评分口径一致。
            boxes = labs[k]
            name = f'{meas}_{fid:06d}' if not is_val else f'{fid:06d}'
            pts.tofile(os.path.join(out_dir, name + '.bin'))
            with open(os.path.join(out_dir, name + '.json'), 'w') as f:
                json.dump(boxes, f)
            n_ok += 1

        key = 'val' if is_val else 'train'
        total[key] += n_ok
        print(f'  {meas}: {n_ok} 帧 -> {"val_real" if is_val else "real"}')

    print(f'完成：验证集 {total["val"]} 帧，真实训练集 {total["train"]} 帧')
    print(f'  {C.VAL_REAL}')
    print(f'  {C.TRAIN_REAL}')


if __name__ == '__main__':
    main()
