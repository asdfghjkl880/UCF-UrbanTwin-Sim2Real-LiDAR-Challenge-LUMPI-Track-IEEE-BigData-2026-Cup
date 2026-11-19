"""把 build_synthbg 产出的 shard 目录展开成提交用的逐帧 .bin（N×3 float32）。

提交包要求 synthetic/ 下恰好 50 个 .bin，每帧 N×3 float32、≤1.5 MB。
shard 目录是 shard_XXX.npy + shard_XXX_meta.json（含 offsets）的打包格式，
这里按 offsets 切回单帧并写裸 float32。

用法:
  py scripts/shard2bin.py --src data/train/sbfinal --out data/synthetic_v14 --n 50
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.out, "*.bin")):
        os.remove(f)

    frames = []
    for sh in sorted(glob.glob(os.path.join(a.src, "shard_*.npy"))):
        pts = np.load(sh)
        meta = json.load(open(sh[:-4] + "_meta.json"))
        off = meta["offsets"]
        for i in range(len(off) - 1):
            frames.append(pts[off[i]:off[i + 1]])
            if len(frames) >= a.n:
                break
        if len(frames) >= a.n:
            break

    if len(frames) < a.n:
        raise SystemExit(f"只有 {len(frames)} 帧，需要 {a.n}")

    sizes = []
    for k, p in enumerate(frames[:a.n]):
        p = np.ascontiguousarray(p, np.float32)
        fn = os.path.join(a.out, f"synthetic_{k:04d}.bin")
        p.tofile(fn)
        sizes.append(os.path.getsize(fn))
    print(f"写出 {a.n} 帧 -> {a.out}")
    print(f"  点数 {min(len(p) for p in frames[:a.n]):,}~"
          f"{max(len(p) for p in frames[:a.n]):,}  "
          f"字节 {min(sizes)/1024:.0f}~{max(sizes)/1024:.0f} KB")


if __name__ == "__main__":
    main()
