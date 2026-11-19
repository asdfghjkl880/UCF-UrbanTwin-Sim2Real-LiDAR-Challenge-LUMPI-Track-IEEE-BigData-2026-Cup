"""把分片数据集按 shard 切成 train/val 两个目录，用硬链接，不额外占磁盘。

用法：
    python scripts/split_shards.py --src data/train/synth3 \
        --train data/train/synth3_tr --val data/train/synth3_va --n-val 4
"""
import argparse
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--n-val', type=int, default=4,
                    help='末尾多少个分片划给验证集')
    a = ap.parse_args()

    stems = sorted(f[:-4] for f in os.listdir(a.src)
                   if f.startswith('shard_') and f.endswith('.npy'))
    if not stems:
        raise SystemExit(f'{a.src} 里没有分片')
    if a.n_val >= len(stems):
        raise SystemExit(f'验证集分片数 {a.n_val} 不小于总分片数 {len(stems)}')

    tr, va = stems[:-a.n_val], stems[-a.n_val:]
    for d in (a.train, a.val):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    def link(stem, dst):
        for ext in ('.npy', '_meta.json'):
            s = os.path.join(a.src, stem + ext)
            t = os.path.join(dst, stem + ext)
            if os.path.exists(t):
                os.remove(t)
            try:
                os.link(s, t)          # 硬链接：同一 inode，不复制数据
            except OSError:
                shutil.copy2(s, t)     # 跨卷时退回复制

    for s in tr:
        link(s, a.train)
    for s in va:
        link(s, a.val)

    sz = sum(os.path.getsize(os.path.join(a.src, s + '.npy')) for s in tr)
    print(f'训练 {len(tr)} 片 / 验证 {len(va)} 片（{va[0]} … {va[-1]}）')
    print(f'训练集约 {sz / 2 ** 30:.1f} GB（硬链接，不新增占用）')


if __name__ == '__main__':
    sys.exit(main())
