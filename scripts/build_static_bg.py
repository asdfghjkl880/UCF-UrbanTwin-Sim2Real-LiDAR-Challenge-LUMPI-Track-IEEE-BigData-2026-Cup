"""零标签静态背景体素场。

只用 M4 合法帧（排除 forbidden_frames.txt 的 100 帧）的**原始点云**，
统计每个体素在多少帧中出现过（时间持续度），从而把
  长期恒定的结构（地面 / 建筑立面 / 树干树冠 / 灯杆 / 护栏）
和
  瞬态的前景（车 / 人 / 骑行者 / 临时堆放物）
分开。

全程不读取任何 Label.csv / GT 框，满足 "no real labels may enter the
training pipeline"。

输出 data/derived/static_bg.npz：
  keys    (M,)   int32   被占用体素的线性编号（稀疏存储）
  counts  (M,)   int16   该体素出现的帧数
  rep     (M,3)  float32 该体素首次出现时的**真实点坐标**（不是格点）
  shape   (3,)           网格尺寸 (NX, NY, NZ)
  meta    json           体素边长 / 原点 / 总帧数 / 帧号清单

用法：
  python scripts/build_static_bg.py --frames 300            # 验证分布
  python scripts/build_static_bg.py --frames 0 --window      # 窗口内全量
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


def forbidden_set():
    s = set()
    with open(FORBIDDEN, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                s.add(ln)
    return s


def legal_frames(excl=None):
    """返回全部合法帧号（字典序升序），已剔除禁用帧。

    excl 用于额外排除**评估帧**：holdout 的 133 帧若也进入背景池，模型就等于
    在验证帧的背景几何上训练过，holdout 指标会虚高。训练侧必须把它们排除，
    评估侧才能反映真实泛化。
    """
    bad = forbidden_set()
    if excl:
        bad |= set(excl)
    out = []
    for fn in sorted(os.listdir(LIDAR)):
        if not fn.endswith(".ply"):
            continue
        stem = fn[:-4]
        if stem in bad:
            continue
        out.append(stem)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300,
                    help="抽样帧数；0 = 用全部合法帧")
    ap.add_argument("--window", type=int, default=1,
                    help="1 = 只用 006497~008109 时间窗内的合法帧（与测试集同段）")
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--lim", type=float, default=45.0, help="x/y 半宽")
    ap.add_argument("--zlo", type=float, default=-5.0)
    ap.add_argument("--zhi", type=float, default=5.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/derived/static_bg.npz"))
    ap.add_argument("--excl-dir", default="",
                    help="排除该目录下 .bin 对应的帧号（通常是 holdout，防背景泄漏）")
    a = ap.parse_args()

    v = a.voxel
    x0 = y0 = -a.lim
    z0 = a.zlo
    NX = int(round(2 * a.lim / v))
    NY = NX
    NZ = int(round((a.zhi - a.zlo) / v))
    NV = NX * NY * NZ
    print(f"网格 {NX}x{NY}x{NZ} = {NV/1e6:.2f}M 体素 @ {v} m", flush=True)

    excl = set()
    if a.excl_dir and os.path.isdir(a.excl_dir):
        excl = {f[:-4] for f in os.listdir(a.excl_dir) if f.endswith(".bin")}
        print(f"排除评估帧 {len(excl)}", flush=True)
    fr = legal_frames(excl)
    if a.window:
        fr = [f for f in fr if "006497" <= f <= "008109"]
    print(f"合法帧 {len(fr)}（时间窗过滤={'开' if a.window else '关'}）", flush=True)
    if a.frames and a.frames < len(fr):
        sel = np.linspace(0, len(fr) - 1, a.frames).astype(int)
        fr = [fr[i] for i in sel]
        print(f"抽样至 {len(fr)} 帧", flush=True)

    counts = np.zeros(NV, np.int16)
    rep = np.zeros((NV, 3), np.float32)
    n_used = 0
    t0 = time.time()
    for j, stem in enumerate(fr):
        p = os.path.join(LIDAR, stem + ".ply")
        if not os.path.exists(p):
            continue
        try:
            _, xyz = M.load_frame(p)
        except Exception:
            continue
        xyz = xyz.astype(np.float32)
        m = ((np.abs(xyz[:, 0]) <= a.lim) & (np.abs(xyz[:, 1]) <= a.lim)
             & (xyz[:, 2] >= a.zlo) & (xyz[:, 2] < a.zhi))
        xyz = xyz[m]
        if len(xyz) < 100:
            continue
        ix = ((xyz[:, 0] - x0) / v).astype(np.int32)
        iy = ((xyz[:, 1] - y0) / v).astype(np.int32)
        iz = ((xyz[:, 2] - z0) / v).astype(np.int32)
        np.clip(ix, 0, NX - 1, out=ix)
        np.clip(iy, 0, NY - 1, out=iy)
        np.clip(iz, 0, NZ - 1, out=iz)
        k = (ix * NY + iy) * NZ + iz
        # 一帧内同一体素只计一次
        uk, first = np.unique(k, return_index=True)
        new = counts[uk] == 0
        if new.any():
            rep[uk[new]] = xyz[first[new]]
        counts[uk] += 1
        n_used += 1

        if (j + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {j+1}/{len(fr)}  已用 {n_used} 帧  "
                  f"{el:.0f}s  预计剩余 {el/(j+1)*(len(fr)-j-1):.0f}s", flush=True)

    nz = np.flatnonzero(counts > 0)
    occ = counts[nz].astype(np.float32) / max(n_used, 1)
    print(f"\n完成：{n_used} 帧，占用体素 {len(nz)/1e6:.3f}M "
          f"（占网格 {100*len(nz)/NV:.2f}%）", flush=True)
    hist = np.histogram(occ, bins=[0, .01, .05, .1, .2, .3, .5, .7, .9, 1.001])[0]
    edges = [0, .01, .05, .1, .2, .3, .5, .7, .9, 1.0]
    print("占用率分布：")
    for i in range(len(hist)):
        print(f"  [{edges[i]:.2f}, {edges[i+1]:.2f})"
              f"  {hist[i]:>9d}  {100*hist[i]/len(nz):5.1f}%")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        keys=nz.astype(np.int32),
        counts=counts[nz].astype(np.int16),
        rep=rep[nz].astype(np.float32),
        shape=np.array([NX, NY, NZ], np.int32),
        meta=json.dumps(dict(voxel=v, x0=x0, y0=y0, z0=z0, lim=a.lim,
                             zlo=a.zlo, zhi=a.zhi, n_frames=n_used,
                             window=bool(a.window), frames=fr)),
    )
    print(f"已写 {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
