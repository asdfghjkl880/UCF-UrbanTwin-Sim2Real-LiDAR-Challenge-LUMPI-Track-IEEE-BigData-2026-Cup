#!/usr/bin/env python3
"""
多进程批量生成合成训练集（分片存储）。

为什么分片：这台机器的 Windows Defender 对每个小文件做实时扫描，实测删除
一个小文件要 ~200ms，一帧两个文件（.bin + .json）意味着 4000 帧要 27 分钟
纯文件开销，生成速度被拖到 1.5 帧/s。改成每片一个大 .npy + 一个 meta.json，
文件数从 8000 降到 32，速度恢复到 10 帧/s 以上，DataLoader 也能 mmap。

输出：
    data/train/synth/shard_000.npy        float32 (总点数, 3)，可 mmap
    data/train/synth/shard_000_meta.json  {"offsets": [...], "labels": [[[cls,x,y,z,L,W,H,hd],...],...]}

用法：
    python scripts/build_trainset.py --n 4000 --shard 250 --workers 16
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402


def _work(task):
    """task = (shard_idx, [(seed,), ...]) -> (shard_idx, pts (S,3), offsets, labels)"""
    from scene_gen import gen_frame
    shard_idx, seeds = task
    chunks, offsets, labels = [], [0], []
    for seed in seeds:
        pts, boxes = gen_frame(seed=seed)
        chunks.append(pts.astype(np.float32))
        offsets.append(offsets[-1] + len(pts))
        labels.append([[b['class_id']] + [round(float(v), 3) for v in b['box']]
                       for b in boxes])
    return shard_idx, np.vstack(chunks), offsets, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--shard', type=int, default=250)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=C.TRAIN_SYNTH,
                    help='输出目录（默认 data/train/synth；该目录内文件会被清空）')
    a = ap.parse_args()

    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):      # 分片文件很少，删除开销可以接受
        os.remove(os.path.join(out_dir, f))

    n_shard = (a.n + a.shard - 1) // a.shard
    tasks = []
    for s in range(n_shard):
        seeds = [a.seed * 1000003 + s * a.shard + i
                 for i in range(min(a.shard, a.n - s * a.shard))]
        tasks.append((s, seeds))

    t0 = time.time()
    done = 0
    with mp.Pool(a.workers) as pool:
        for si, pts, offs, labels in pool.imap_unordered(_work, tasks):
            stem = os.path.join(out_dir, f"shard_{si:03d}")
            np.save(stem + '.npy', pts)
            with open(stem + '_meta.json', 'w') as f:
                json.dump({'offsets': offs, 'labels': labels}, f)
            done += 1
            el = time.time() - t0
            print(f'  片 {done}/{n_shard}  {len(pts):,} 点  '
                  f'{el:.0f}s  {done*len(tasks[0][1])/el:.1f} 帧/s', flush=True)

    el = time.time() - t0
    tot = sum(os.path.getsize(os.path.join(out_dir, f))
              for f in os.listdir(out_dir))
    print(f'完成 {a.n} 帧 / {n_shard} 片，{el:.1f}s（{a.n/el:.1f} 帧/s），'
          f'{tot/2**30:.1f} GB')
    print(f'  -> {C.TRAIN_SYNTH}')


if __name__ == '__main__':
    mp.freeze_support()
    main()
