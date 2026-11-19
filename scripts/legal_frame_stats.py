"""按主办方 2026-11-11 澄清口径，只用**合法真实帧**统计类别尺寸/数量先验。

规则依据（rules.md 第 9/11 行 + admins 消息）：
  * 禁用集 = forbidden_frames.txt 里那 100 个 ID，**不是**整个 6497~8109 区间；
  * 剔除这 100 帧后，其余真实帧可用于 validation / model selection /
    box-size statistics（允许读标签）/ 合成数据 realism 标定；
  * 检测器本身只能用合成数据训练。

用法：
  python scripts/legal_frame_stats.py                      # 全部合法帧
  python scripts/legal_frame_stats.py --window 6497 8109   # 只看测试时段
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

import paths

CLASS_NAMES = {0: 'Person', 1: 'Car', 2: 'Bicycle', 3: 'Motorcycle',
               4: 'Bus', 5: 'Truck', 6: 'Van', 7: 'Unknown'}


def _read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def load_forbidden():
    p = os.path.join(paths.KIT, 'forbidden_frames.txt')
    ids = set()
    for line in _read(p).splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            ids.add(int(line))
    return ids


def load_labels():
    """返回 {frame_idx: [(cls, L, W, H, cx, cy, cz, hd), ...]}"""
    p = os.path.join(paths.LABEL_DIR, 'Measurement4', 'Label.csv')
    frames = defaultdict(list)
    with open(p, newline='', encoding='utf-8', errors='replace') as f:
        rd = csv.reader(f)
        head = next(rd)
        cols = [c.strip() for c in head]
        ci = {name: i for i, name in enumerate(cols)}
        i_t, i_c = ci['time'], ci['class_id']
        base = ci['3D box: center x']   # x,y,z,L,W,H,heading 连续排布
        for row in rd:
            if len(row) < base + 7:
                continue
            try:
                t = float(row[i_t])
                cls = int(float(row[i_c]))
                x, y, z, L, W, H, hd = (float(row[base + k]) for k in range(7))
            except ValueError:
                continue
            frames[int(round(t * 10))].append((cls, L, W, H, x, y, z, hd))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', nargs=2, type=int, default=None,
                    help='只统计 [lo, hi] 帧号区间（仍剔除禁用帧）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    forbidden = load_forbidden()
    frames = load_labels()
    all_ids = sorted(frames)
    legal = [f for f in all_ids if f not in forbidden]

    sel = legal
    if a.window:
        lo, hi = a.window
        sel = [f for f in legal if lo <= f <= hi]

    print('M4 总帧 %d | 禁用 %d | 合法 %d | 本次统计 %d'
          % (len(all_ids), len(forbidden), len(legal), len(sel)))
    if a.window:
        nfb = sum(1 for f in forbidden if a.window[0] <= f <= a.window[1])
        print('  （区间 %d~%d，已剔除区间内 %d 个禁用帧）'
              % (a.window[0], a.window[1], nfb))

    size = defaultdict(list)
    for f in sel:
        for (cls, L, W, H, *_rest) in frames[f]:
            if cls == 7 or cls not in CLASS_NAMES:
                continue
            size[cls].append((L, W, H))

    n = max(len(sel), 1)
    out = {'n_frames': len(sel), 'classes': {}}
    print('')
    print('%-12s%7s%8s   %9s%9s%9s%8s'
          % ('class', 'n_box', 'per_frm', 'L_med', 'W_med', 'H_med', 'L_p90'))
    tot = 0
    for cls in sorted(size):
        arr = np.array(size[cls], dtype=np.float64)
        med = np.median(arr, axis=0)
        p90 = np.percentile(arr[:, 0], 90)
        per = len(arr) / n
        tot += len(arr)
        print('%-12s%7d%8.2f   %9.2f%9.2f%9.2f%8.2f'
              % (CLASS_NAMES[cls], len(arr), per, med[0], med[1], med[2], p90))
        out['classes'][CLASS_NAMES[cls]] = {
            'n': len(arr), 'per_frame': round(per, 3),
            'LWH_median': [round(float(v), 3) for v in med],
        }
    print('%-12s%7d%8.2f' % ('TOTAL', tot, tot / n))
    out['total_per_frame'] = round(tot / n, 3)

    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print('\nwritten ' + a.out)


if __name__ == '__main__':
    main()
