"""从合法无标签帧提取稠密静态背景点云池。

两级体素：
  粗 0.2 m  由 build_static_bg.py 统计时间持续度，占用率 >= --thr 判为静态结构
  细 0.06 m 只在静态粗体素内部采样，保留真实测量点（不是格点），去重后得到
            稠密的静态表面点云

全程不读任何 GT / Label.csv。
输出 data/derived/bgpool.npy  (N,3) float32
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m4_sensor as M  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIDAR = os.path.join(ROOT, "data/LUMPI/Measurement4/lidar")
FORBIDDEN = os.path.join(ROOT, "lumpi_starting_kit/forbidden_frames.txt")


def legal_frames(lo=None, hi=None):
    bad = set()
    with open(FORBIDDEN, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                bad.add(ln)
    out = []
    for fn in sorted(os.listdir(LIDAR)):
        if not fn.endswith(".ply"):
            continue
        s = fn[:-4]
        if s in bad:
            continue
        if lo and hi and not (lo <= s <= hi):
            continue
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default=os.path.join(ROOT, "data/derived/static_bg_w.npz"))
    ap.add_argument("--thr", type=float, default=0.7, help="粗体素静态阈值")
    ap.add_argument("--fine", type=float, default=0.06, help="细去重边长")
    ap.add_argument("--batch", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/derived/bgpool.npy"))
    a = ap.parse_args()

    z = np.load(a.bg, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    NX, NY, NZ = [int(t) for t in z["shape"]]
    v = meta["voxel"]
    occ = z["counts"].astype(np.float32) / max(meta["n_frames"], 1)
    good = z["keys"][occ >= a.thr]
    mask = np.zeros(NX * NY * NZ, np.uint8)
    mask[good] = 1
    print(f"静态粗体素 {len(good)/1e3:.1f}k / {NX*NY*NZ/1e6:.2f}M  (thr={a.thr})",
          flush=True)

    fr = legal_frames("006497", "008109")
    print(f"合法帧 {len(fr)}", flush=True)

    fv = a.fine
    fkeys, fpts = [], []
    t0 = time.time()
    n_keep = 0
    for bi in range(0, len(fr), a.batch):
        chunk = fr[bi:bi + a.batch]
        ck, cp = [], []
        for stem in chunk:
            p = os.path.join(LIDAR, stem + ".ply")
            if not os.path.exists(p):
                continue
            try:
                _, xyz = M.load_frame(p)
            except Exception:
                continue
            xyz = xyz.astype(np.float32)
            m = ((np.abs(xyz[:, 0]) <= meta["lim"]) & (np.abs(xyz[:, 1]) <= meta["lim"])
                 & (xyz[:, 2] >= meta["zlo"]) & (xyz[:, 2] < meta["zhi"]))
            xyz = xyz[m]
            if len(xyz) < 100:
                continue
            ix = ((xyz[:, 0] - meta["x0"]) / v).astype(np.int32)
            iy = ((xyz[:, 1] - meta["y0"]) / v).astype(np.int32)
            iz = ((xyz[:, 2] - meta["z0"]) / v).astype(np.int32)
            np.clip(ix, 0, NX - 1, out=ix)
            np.clip(iy, 0, NY - 1, out=iy)
            np.clip(iz, 0, NZ - 1, out=iz)
            k = (ix * NY + iy) * NZ + iz
            keep = mask[k] == 1
            if not keep.any():
                continue
            q = xyz[keep]
            n_keep += len(q)
            fk = (np.floor((q[:, 0] - meta["x0"]) / fv).astype(np.int64) * 4001
                  + np.floor((q[:, 1] - meta["y0"]) / fv).astype(np.int64)) * 2731 \
                + np.floor((q[:, 2] - meta["z0"]) / fv).astype(np.int64)
            ck.append(fk)
            cp.append(q)
        if ck:
            allk = np.concatenate(ck)
            allp = np.concatenate(cp)
            uk, first = np.unique(allk, return_index=True)
            fkeys.append(uk)
            fpts.append(allp[first])
        el = time.time() - t0
        print(f"  {min(bi+a.batch,len(fr))}/{len(fr)}  累计保留 {n_keep/1e6:.2f}M 点次  "
              f"{el:.0f}s", flush=True)

    allk = np.concatenate(fkeys)
    allp = np.concatenate(fpts)
    o = np.argsort(allk, kind="stable")
    allk, allp = allk[o], allp[o]
    uniq = np.r_[True, allk[1:] != allk[:-1]]
    pool = allp[uniq].astype(np.float32)
    print(f"\n去重后背景池 {len(pool)/1e6:.3f}M 点  "
          f"范围 x[{pool[:,0].min():.1f},{pool[:,0].max():.1f}] "
          f"y[{pool[:,1].min():.1f},{pool[:,1].max():.1f}] "
          f"z[{pool[:,2].min():.1f},{pool[:,2].max():.1f}]")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.save(a.out, pool)
    print(f"已写 {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
