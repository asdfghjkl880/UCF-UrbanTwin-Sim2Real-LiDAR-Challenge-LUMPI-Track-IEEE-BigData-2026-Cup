#!/usr/bin/env python3
"""
闭环标定 scene_gen.POINT_K：让合成帧里"框内实际点数 × r^2"的中位数
对上真实帧的实测值（Person 17183 / Car 114501 / …）。

为什么要闭环：POINT_K 是采样时的目标值，但采样后还有两道损耗——
  * `drop=0.15` 随机丢点
  * 双向遮挡 `RangeGrid.occluded` 会把被背景挡住的目标点删掉
两类目标的损耗比例不同（行人比车更容易被地面挡住），所以不能用一个全局系数补。

    python scripts/calib_points.py --n 40
输出建议的 POINT_K，手工填回 scene_gen.py（或加 --apply 自动改）。
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as G  # noqa: E402

# 真实帧实测（scripts/diag_points.py 在 M4 31 帧真实帧上测得，r<=40m 的框，
# 口径与 measure() 一致：框内点数 × r²，取中位数）。Bus/Truck/Unknown 在真实帧里
# 全部落在 40m 之外（框内 0 点），无法标定，保持原值。
TARGET = {'Person': 8612, 'Car': 79229, 'Bicycle': 17063}


def measure(n_frames, seed=0):
    acc = defaultdict(list)
    for i in range(n_frames):
        pts, boxes = G.gen_frame(seed=seed * 1000 + i)
        for b in boxes:
            cx, cy, cz, L, W, H, hd = b['box']
            r = float(np.hypot(cx, cy))
            # 口径必须与 diag_points.py（TARGET 的来源）完全一致：
            # 严格框内、不放大边距、只统计 1<=r<=40 的框。否则同一批数据能差 2 倍。
            if not (1.0 <= r <= 40.0):
                continue
            dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
            c, s = np.cos(hd), np.sin(hd)
            u = dx * c + dy * s
            v = -dx * s + dy * c
            m = ((np.abs(u) <= L / 2) & (np.abs(v) <= W / 2) &
                 (np.abs(pts[:, 2] - cz) <= H / 2))
            acc[b['class']].append(int(m.sum()) * r * r)
    return {k: float(np.median(v)) for k, v in acc.items() if len(v) >= 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()

    got = measure(a.n)
    print(f'{"类别":<11}{"当前 K":>10}{"实测 n*r^2":>12}{"目标":>10}{"比值":>8}{"建议 K":>11}')
    newk = dict(G.POINT_K)
    for k in G.POINT_K:
        cur = G.POINT_K[k]
        g = got.get(k)
        t = TARGET.get(k)
        if g is None or t is None:
            print(f'{k:<11}{cur:>10}{"—":>12}'
                  f'{(t if t else "—"):>10}{"—":>8}{cur:>11}')
            continue
        sug = int(round(cur * t / max(g, 1.0)))
        newk[k] = sug
        print(f'{k:<11}{cur:>10}{g:>12.0f}{t:>10}{g/t:>8.2f}{sug:>11}')

    if a.apply:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scene_gen.py')
        s = open(p, encoding='utf-8').read()
        old = s[s.index('POINT_K = {'):s.index('}', s.index('POINT_K = {')) + 1]
        items = ', '.join(f"'{k}': {newk[k]}" for k in G.POINT_K)
        new = 'POINT_K = {' + items + '}'
        s = s.replace(old, new)
        open(p, 'w', encoding='utf-8').write(s)
        print('\n已写回 scene_gen.py：')
        print('  ' + new)


if __name__ == '__main__':
    main()
