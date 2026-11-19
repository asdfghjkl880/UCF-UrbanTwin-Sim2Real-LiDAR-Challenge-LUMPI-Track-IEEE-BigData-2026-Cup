"""两遍标定：实测合成端"最终可见点数 / 请求点数"的比值，按类写回修正系数。

第一遍按 _target_count 请求点数，生成若干帧后统计框内实际点数；
比值 < 1 说明还有损耗（背景遮挡、框边裁剪、目标互相遮挡），需要放大请求量。
结果写入 reports/obj_density_cal.json，scene_gen 启动时自动叠加。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ['OBJ_DENSITY_CAL'] = '0'
os.environ['OBJ_DENSITY_KEEP'] = '1.0'
import scene_gen as G  # noqa: E402

BINS = [(6, 10), (10, 14), (14, 20), (20, 26), (26, 33), (33, 41)]


def in_box_count(pts, b, z_floor=-1.6):
    cx, cy, cz, L, W, H, hd = b
    c, s = np.cos(-hd), np.sin(-hd)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    m = ((np.abs(lx) <= L / 2) & (np.abs(ly) <= W / 2) &
         (pts[:, 2] >= max(cz - H / 2, z_floor)) & (pts[:, 2] <= cz + H / 2))
    return int(m.sum())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--out', default='reports/obj_density_cal.json')
    a = ap.parse_args()

    tab = json.load(open(G.OBJ_DENSITY_JSON))
    acc = {}
    for i in range(a.n):
        rng_seed = 730000 + i
        rng = np.random.default_rng(rng_seed)
        objs = G.make_objects(rng, G.get_lanes())
        want = np.array([G._target_count(G.CLASSES[o[0]],
                                         float(np.hypot(o[1], o[2])))
                         for o in objs])
        pts, boxes = G.gen_frame(seed=rng_seed)
        for o, w, bx in zip(objs, want, boxes):
            b = bx['box']
            c = G.CLASSES[o[0]]
            m = in_box_count(pts, b)
            acc.setdefault(c, []).append((w, m, float(np.hypot(b[0], b[1]))))

    cal = {}
    print(f'{"class":<11}{"请求中位":>10}{"实测中位":>10}{"比值":>8}{"n":>7}')
    for c, rows in acc.items():
        w = np.array([x[0] for x in rows])
        m = np.array([x[1] for x in rows], float)
        if len(rows) < 10:
            continue
        ratio = float(np.median(m) / max(np.median(w), 1e-6))
        cal[c] = round(1.0 / max(ratio, 0.2), 3)
        print(f'{c:<11}{np.median(w):>10.1f}{np.median(m):>10.1f}'
              f'{ratio:>8.2f}{len(rows):>7}')
    json.dump(cal, open(a.out, 'w'), indent=1)
    print(f'\n-> {a.out}  {cal}')


if __name__ == '__main__':
    main()
