"""合成域 vs 真实域：目标点占比 + 每类框内 n·r² —— 用于闭环标定 POINT_K。

用法:
    py scripts/diag_points.py            # 测当前 scene_gen 生成的 30 帧
    py scripts/diag_points.py --shard 0  # 测已落盘的 shard_000
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C


def rot(p, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([p[:, 0] * c + p[:, 1] * s, -p[:, 0] * s + p[:, 1] * c], 1)


def in_box(pts, b):
    d = pts - np.array([b[1], b[2], b[3]], np.float32)
    z = np.abs(d[:, 2]) <= b[6] / 2.0
    p = rot(d[:, :2], -b[7])
    return z & (np.abs(p[:, 0]) <= b[4] / 2.0) & (np.abs(p[:, 1]) <= b[5] / 2.0)


def summarize(frames, tag):
    """frames: [(pts (N,3), boxes [ [cls,cx,cy,cz,L,W,H,hd], ... ])]"""
    KR = collections.defaultdict(list)
    nobj = collections.Counter()
    obj_frac = []
    for pts, boxes in frames:
        m = np.zeros(len(pts), bool)
        for b in boxes:
            k = in_box(pts, b)
            m |= k
            n = int(k.sum())
            r = float(np.hypot(b[1], b[2]))
            nm = C.CLASS_NAMES[int(b[0])]
            nobj[nm] += 1
            if 1 <= r <= 40:
                KR[nm].append(n * r * r)
        obj_frac.append(m.mean())

    nf = len(frames)
    print(f'\n===== {tag}（{nf} 帧）=====')
    print(f'  目标点占全帧比例  {np.mean(obj_frac)*100:5.2f}%  '
          f'(±{np.std(obj_frac)*100:.2f})')
    print(f'  {"class":<12}{"框/帧":>7}{"n·r² p50":>11}{"n·r² p25":>11}{"n·r² p75":>11}')
    for nm in C.CLASS_NAMES:
        a = np.array(KR[nm], np.float64)
        if a.size == 0:
            print(f'  {nm:<12}{nobj[nm]/nf:>7.1f}{"-":>11}{"-":>11}{"-":>11}')
            continue
        print(f'  {nm:<12}{nobj[nm]/nf:>7.1f}{np.percentile(a,50):>11.0f}'
              f'{np.percentile(a,25):>11.0f}{np.percentile(a,75):>11.0f}')
    return {nm: np.percentile(np.array(v), 50) if v else 0.0 for nm, v in KR.items()}, \
        float(np.mean(obj_frac))


def load_real():
    out = []
    for fn in sorted(os.listdir(C.VAL_REAL)):
        if not fn.endswith('.bin'):
            continue
        pts = np.fromfile(os.path.join(C.VAL_REAL, fn), np.float32).reshape(-1, 3)
        out.append((pts, json.load(open(os.path.join(C.VAL_REAL, fn[:-4] + '.json')))))
    return out


def load_shard(k, n=30):
    stem = os.path.join(C.TRAIN_SYNTH, f'shard_{k:03d}')
    arr = np.load(stem + '.npy', mmap_mode='r')
    meta = json.load(open(stem + '_meta.json'))
    offs, labs = meta['offsets'], meta['labels']
    out = []
    for i in range(min(n, len(labs))):
        out.append((np.asarray(arr[offs[i]:offs[i + 1]], np.float32), labs[i]))
    return out


def gen_fresh(n=30):
    import scene_gen as S
    out = []
    for i in range(n):
        pts, boxes = S.gen_frame(seed=90000 + i)
        out.append((pts, [[S.CLASSES.index(b['class'])] + b['box'] for b in boxes]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()

    real_k, real_f = summarize(load_real(), '真实 M4 验证帧')
    if a.shard >= 0:
        syn_k, syn_f = summarize(load_shard(a.shard, a.n), f'已落盘 shard_{a.shard:03d}')
    else:
        syn_k, syn_f = summarize(gen_fresh(a.n), '当前 scene_gen 新生成')

    print('\n----- 闭环系数（真实 / 合成，>1 表示合成太稀，POINT_K 要乘这个数）-----')
    for nm in C.CLASS_NAMES:
        if syn_k.get(nm, 0) > 0 and real_k.get(nm, 0) > 0:
            print(f'  {nm:<12}{real_k[nm]/syn_k[nm]:6.2f}')
    print(f'\n  目标点占比  真实 {real_f*100:.2f}%  vs  合成 {syn_f*100:.2f}%  '
          f'-> 系数 {real_f/max(syn_f,1e-9):.2f}')


if __name__ == '__main__':
    main()
