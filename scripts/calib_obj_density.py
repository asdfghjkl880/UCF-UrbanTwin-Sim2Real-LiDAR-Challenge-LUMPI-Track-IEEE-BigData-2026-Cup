#!/usr/bin/env python3
"""标定 POINT_K：让合成帧的"框内点数"分布对齐真实测试分布。

为什么需要单独标定：POINT_K 决定的是**请求**点数，经过 15% drop、双向遮挡、
整帧 5 万点预算之后剩下的才是真正进框的点。这条链路上每一环都会打折，
所以只能"测量-反馈"地解，不能解析反推。

2026-11-10 的教训：旧标定收敛到 1.1% 残差，但用的是**错误的口径**
（圆柱近似，把地面杂波算进了目标点），实际合成框内点中位 8 个、真实 76 个，
差 10 倍。本脚本与 probe 用完全相同的口径，两侧对称测量。

口径：
  框内点 = 在 GT 长方体内部 且 z > -1.6（排除框底切到的地面点）。
  真实框底普遍切到地面（Car 的 cz=-1.33, H=1.62 → 底部 -2.14，地面 -2.05），
  不排掉地面点会把 Car 的 96 点里约一半算成"目标回波"。

用法：
    python scripts/calib_obj_density.py --rounds 3 --frames 60
    python scripts/calib_obj_density.py --write     # 收敛后写回 scene_gen.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as G  # noqa: E402

CLS = G.CLASSES
Z_AIR = -1.60
MIN_BOXES = 40          # 样本太少的类不参与标定（Bus 在测试集只有 5 个框）


def in_box(pts, b):
    cx, cy, cz, L, W, H, hd = b
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    c, s = np.cos(-hd), np.sin(-hd)
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    return ((np.abs(lx) <= L / 2) & (np.abs(ly) <= W / 2) &
            (np.abs(pts[:, 2] - cz) <= H / 2) & (pts[:, 2] > Z_AIR))


def measure(frames):
    """frames: [(pts, boxes[n,8])] -> {cls: [n_in_box,...]}, 目标点占整帧比例"""
    per = {k: [] for k in CLS}
    frac = []
    for pts, boxes in frames:
        anym = np.zeros(len(pts), bool)
        for b in boxes:
            ci = int(b[0])
            if ci < 0 or ci >= len(CLS):
                continue
            m = in_box(pts, b[1:8])
            per[CLS[ci]].append(int(m.sum()))
            anym |= m
        frac.append(anym.mean())
    return per, float(np.mean(frac))


def real_frames(n):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, 'data', 'eval_m4_test', 'detection')
    out = []
    for f in sorted(glob.glob(os.path.join(d, '*.bin')))[:n]:
        pts = np.fromfile(f, np.float32).reshape(-1, 3)
        with open(f[:-4] + '.json') as fh:
            out.append((pts, [np.asarray(b, np.float32) for b in json.load(fh)]))
    return out


def synth_frames(n, seed0=90000):
    out = []
    bg = G.get_background()
    for i in range(n):
        pts, boxes = G.gen_frame(seed=seed0 + i, background=bg, align='translate')
        rows = (np.array([[b['class_id']] + list(b['box']) for b in boxes], np.float32)
                if boxes else np.zeros((0, 8), np.float32))
        out.append((pts, rows.reshape(-1, 8)))
    return out


def med(per):
    return {k: (float(np.median(v)) if len(v) >= MIN_BOXES else None)
            for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    rt = real_frames(a.frames)
    rper, rfrac = measure(rt)
    rtgt = med(rper)
    print('真实目标（%d 帧，框内且 z>%.2f）：' % (a.frames, Z_AIR))
    for k in CLS:
        n = len(rper[k])
        print('  %-11s n_box %5d  中位 %s' % (
            k, n, ('%6.1f' % rtgt[k]) if rtgt[k] is not None else '  n/a'))
    print('  框内点占整帧 %.3f%%' % (100 * rfrac))

    K = dict(G.POINT_K)
    for r in range(a.rounds):
        G.POINT_K.update(K)
        sf = synth_frames(a.frames)
        sper, sfrac = measure(sf)
        smed = med(sper)
        print('--- round %d  合成框内点占比 %.3f%%' % (r, 100 * sfrac))
        newK = dict(K)
        for k in CLS:
            t, s = rtgt[k], smed[k]
            if t is None or s is None or s <= 0:
                print('  %-11s 跳过（真实 %s / 合成 %s）' % (
                    k, t, None if s is None else round(s, 1)))
                continue
            ratio = t / s
            ratio = float(np.clip(ratio, 0.25, 25.0))
            newK[k] = int(round(K[k] * ratio))
            print('  %-11s 真实 %6.1f  合成 %6.1f  比 %5.2f  K %9d -> %9d'
                  % (k, t, s, ratio, K[k], newK[k]))
        K = newK
        res = [abs(rtgt[k] - smed[k]) / max(rtgt[k], 1e-6)
               for k in CLS if rtgt[k] and smed[k]]
        print('  最大相对残差 %.1f%%' % (100 * max(res) if res else 0))

    print('\n最终 POINT_K = ' + json.dumps(K))
    if a.write:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scene_gen.py')
        src = open(p).read()
        old = 'POINT_K = ' + src.split('POINT_K = ', 1)[1].split('\n', 1)[0]
        new = ('POINT_K = ' + '{' + ', '.join("'%s': %d" % (k, K[k]) for k in CLS) + '}')
        src = src.replace(old, new, 1)
        open(p, 'w').write(src)
        print('已写回 scene_gen.py：\n  ' + new)


if __name__ == '__main__':
    main()
