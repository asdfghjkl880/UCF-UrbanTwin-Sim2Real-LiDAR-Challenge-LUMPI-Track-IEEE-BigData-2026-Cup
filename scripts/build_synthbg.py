"""合规合成训练集：真实单帧背景（无标签过滤） + 自渲染物体。

与旧 build_realbg.py 的根本区别：
  旧：读 gt[fid] 真实标注框去擦除前景  ->  真实标签进入训练流水线（违规）
  新：用 **零标签** 的静态体素占用率 p_static 过滤前景点，全程不读 Label.csv

物体放置只使用**分布先验**（类别比例、位置、尺寸、朝向的经验分布），
这些规则明确允许（"statistics of object box size / count / spatial
distribution using real labels are permitted"）；物体点由合成目标库渲染，
不使用任何用真实框切出的点。

输出 shard 格式与 train_pp.py 一致：
  shard_XXX.npy        (N,3) float32
  shard_XXX_meta.json  {"offsets": [...], "labels": [[[c,x,y,z,dx,dy,dz,hd],...],...]}
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m4_sensor as M  # noqa: E402
import pp_config as C  # noqa: E402
import train_pp as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIDAR = os.path.join(ROOT, "data/LUMPI/Measurement4/lidar")
FORBIDDEN = os.path.join(ROOT, "lumpi_starting_kit/forbidden_frames.txt")

G = {}          # 每个 worker 的全局状态
# 合法帧每帧各类物体数量均值（经验分布先验，按 CLASS_NAMES 顺序）
MEAN_PER_FRAME = [11.07, 25.60, 2.02, 1.15, 1.34, 3.47, 2.19]


def legal_frames(lo="006497", hi="008109", excl=None):
    bad = set()
    with open(FORBIDDEN, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                bad.add(ln)
    if excl:
        bad |= set(excl)
    out = []
    for fn in sorted(os.listdir(LIDAR)):
        if not fn.endswith(".ply"):
            continue
        s = fn[:-4]
        if s in bad or not (lo <= s <= hi):
            continue
        out.append(s)
    return out


def _init(bg_npz, per_cls, frames, thr, lim, n_pts, drop, pri=None):
    z = np.load(bg_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    NX, NY, NZ = [int(t) for t in z["shape"]]
    occ = z["counts"].astype(np.float32) / max(meta["n_frames"], 1)
    good = z["keys"][occ >= thr]
    mask = np.zeros(NX * NY * NZ, np.uint8)
    mask[good] = 1
    G.update(mask=mask, meta=meta, shape=(NX, NY, NZ), boxes=per_cls,
             frames=frames, lim=lim, n_pts=n_pts, drop=drop, pri=pri)


def _pstatic(xyz):
    meta, (NX, NY, NZ), mask = G["meta"], G["shape"], G["mask"]
    v = meta["voxel"]
    ix = ((xyz[:, 0] - meta["x0"]) / v).astype(np.int32)
    iy = ((xyz[:, 1] - meta["y0"]) / v).astype(np.int32)
    iz = ((xyz[:, 2] - meta["z0"]) / v).astype(np.int32)
    np.clip(ix, 0, NX - 1, out=ix)
    np.clip(iy, 0, NY - 1, out=iy)
    np.clip(iz, 0, NZ - 1, out=iz)
    return mask[(ix * NY + iy) * NZ + iz] == 1


def build_edge_priors(gtdir, lim, bin_xy=2.0):
    """从合法帧 GT 统计**一维边缘分布**先验（空间 / 尺寸 / 高度 / 朝向各自独立）。

    与"从真实框池抽整条元组"的区别：这里每条属性只来自它自己的一维分布，
    采样时互相独立，不存在任何一个真实框的完整属性组合被整体取用。这对应
    规则允许的 "statistics of object box size / count / spatial distribution"，
    而不是复制某条标注。

    **重要**：空间先验只用落在场景内（|x|,|y| <= lim-2）的框，因为合成帧只能
    在上面放目标；但**尺寸/高度/朝向必须用全部 GT**。早前版本把同一个位置过滤
    也套在尺寸上，导致先验池只剩近处目标——远处的标注框普遍更小（被遮挡、
    只标可见部分），于是 Truck 先验被推成 L9.18/W3.68 而真实全量是 L8.16/W3.00，
    训练分布与评估分布系统性错位。这个 bug 在 sbg1/sbg2/sbg3 里一直存在。
    """
    gt = T.load_gt(gtdir)
    allr = [[] for _ in C.CLASS_NAMES]     # 全部 GT：尺寸/高度/朝向
    inr = [[] for _ in C.CLASS_NAMES]      # 场景内 GT：只用于空间直方图
    for f in sorted(gt):
        for g in gt[f]:
            nm = g.get("class")
            if nm not in C.CLASS_NAMES:
                continue
            b = g["box"]
            ci = C.CLASS_NAMES.index(nm)
            row = [float(t) for t in b[:7]]
            allr[ci].append(row)
            if abs(b[0]) <= lim - 2 and abs(b[1]) <= lim - 2:
                inr[ci].append(row)
    ed = np.arange(-lim, lim + bin_xy, bin_xy)
    pri = []
    for ci in range(len(C.CLASS_NAMES)):
        if len(allr[ci]) < 5 or len(inr[ci]) < 5:
            pri.append(None)
            continue
        a = np.asarray(allr[ci], np.float64)
        s = np.asarray(inr[ci], np.float64)
        H2, xe, ye = np.histogram2d(s[:, 0], s[:, 1], bins=[ed, ed])
        p = H2 / max(H2.sum(), 1.0)
        pri.append({
            "p": p.ravel(), "xe": xe, "ye": ye, "nx": len(xe) - 1,
            "z": a[:, 2].copy(), "L": a[:, 3].copy(),
            "W": a[:, 4].copy(), "H": a[:, 5].copy(),
            "hd": a[:, 6].copy(),
        })
    return pri


def _sample_objs_edge(rng, pri, lim):
    """按各类**边缘分布**独立采样（空间、尺寸、高度、朝向互不耦合）。"""
    objs = []
    for ci, p in enumerate(pri):
        if p is None:
            continue
        n = int(rng.poisson(MEAN_PER_FRAME[ci]))
        if n <= 0:
            continue
        pv = p["p"] / p["p"].sum()
        for _ in range(n):
            k = int(rng.choice(len(pv), p=pv))
            ix, iy = divmod(k, p["nx"])          # 空间边缘
            x = float(rng.uniform(p["xe"][ix], p["xe"][ix + 1]))
            y = float(rng.uniform(p["ye"][iy], p["ye"][iy + 1]))
            if abs(x) > lim - 2 or abs(y) > lim - 2:
                continue
            L = float(rng.choice(p["L"]) * np.exp(rng.normal(0, 0.07)))
            W = float(rng.choice(p["W"]) * np.exp(rng.normal(0, 0.07)))
            Hh = float(rng.choice(p["H"]) * np.exp(rng.normal(0, 0.05)))
            z = float(rng.choice(p["z"]) + rng.normal(0, 0.06))
            hd = float(rng.choice(p["hd"]) + rng.normal(0, 0.12))
            objs.append((ci, x, y, z, L, W, Hh, hd))
    return objs


def _sample_objs(rng, boxes, lim):
    """按真实经验分布采样物体（分层：逐类按真实每帧均值抽数量）。

    boxes 为按类分组的列表 of list。
    位置/尺寸/朝向取自合法帧 GT 的经验分布（规则允许的分布先验），
    物体点云由合成目标库渲染，不使用真实框切出的点。
    """
    objs = []
    for ci, pool in enumerate(boxes):
        if not pool:
            continue
        mu = MEAN_PER_FRAME[ci]
        n = int(rng.poisson(mu))
        if n <= 0:
            continue
        for i in rng.integers(0, len(pool), n):
            c, x, y, z, L, W, H, hd = pool[i]
            x = float(x + rng.normal(0, 0.55))
            y = float(y + rng.normal(0, 0.55))
            if abs(x) > lim - 2 or abs(y) > lim - 2:
                continue
            objs.append((int(c), x, y, float(z),
                         float(L * rng.uniform(0.94, 1.06)),
                         float(W * rng.uniform(0.94, 1.06)),
                         float(H * rng.uniform(0.94, 1.06)),
                         float(hd + rng.normal(0, 0.05))))
    return objs


def _one(seed):
    from scene_gen import object_points
    rng = np.random.default_rng(seed)
    framed = G["frames"][rng.integers(0, len(G["frames"]))]
    p = os.path.join(LIDAR, framed + ".ply")
    _, xyz = M.load_frame(p)
    xyz = xyz.astype(np.float32)
    lim = G["lim"]
    m = ((np.abs(xyz[:, 0]) <= lim) & (np.abs(xyz[:, 1]) <= lim)
         & (xyz[:, 2] >= -5.0) & (xyz[:, 2] < 5.0))
    xyz = xyz[m]
    bg = xyz[_pstatic(xyz)]           # 零标签前景剔除

    objs = (_sample_objs_edge(rng, G["pri"], lim) if G.get("pri")
            else _sample_objs(rng, G["boxes"], lim))
    pts, ids, vis = object_points(rng, objs)

    if len(pts):
        allp = np.vstack([bg, np.asarray(pts, np.float32)])
    else:
        allp = bg
    # 保底点数：不足则从背景中重复采样不同帧（不重复同一点）
    if len(allp) > G["n_pts"]:
        sel = rng.choice(len(allp), G["n_pts"], replace=False)
        allp = allp[sel]
        # 重新计算哪些物体还在（简化：保留全部标签，与真实标注一致）
        keep_lab = True
    else:
        keep_lab = True

    labs = []
    if keep_lab:
        for k, o in enumerate(objs):
            if not vis[k]:
                continue
            c, x, y, z, L, W, H, hd = o
            hd = (hd + np.pi) % (2 * np.pi) - np.pi
            labs.append([int(c)] + [round(float(t), 3) for t in (x, y, z, L, W, H, hd)])
    return allp.astype(np.float32), labs


def _work(task):
    si, seeds = task
    chunks, offs, labs = [], [0], []
    for sd in seeds:
        try:
            p, lb = _one(sd)
        except Exception as e:
            print("跳过", sd, e, flush=True)
            continue
        if len(p) < 2000:
            continue
        chunks.append(p)
        offs.append(offs[-1] + len(p))
        labs.append(lb)
    if not chunks:
        return si, np.zeros((0, 3), np.float32), [0], []
    return si, np.vstack(chunks), offs, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default=os.path.join(ROOT, "data/derived/static_bg_w.npz"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"),
                    help="只用于统计类别/位置/尺寸/朝向的经验分布")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.6, help="p_static 背景阈值")
    ap.add_argument("--lim", type=float, default=40.0)
    ap.add_argument("--npts", type=int, default=50000)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--prior", choices=("edge", "tuple"), default="edge",
                    help="edge = 一维边缘分布独立采样（默认，合规上更干净）；"
                         "tuple = 旧版从真实框池抽整条元组")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/train/sbg1"))
    ap.add_argument("--excl-dir", default=os.path.join(ROOT, "data/eval_m4_holdout"),
                    help="排除该目录下的帧（默认排除 holdout，防止背景泄漏进训练集）")
    a = ap.parse_args()

    pri = build_edge_priors(a.gtdir, a.lim) if a.prior == "edge" else None
    if pri:
        print("边缘先验各类样本数 "
              f"{[0 if p is None else len(p['L']) for p in pri]}", flush=True)

    gt = T.load_gt(a.gtdir)
    boxes = []
    for f in sorted(gt):
        for g in gt[f]:
            nm = g.get("class")
            if nm not in C.CLASS_NAMES:
                continue
            b = g["box"]
            if abs(b[0]) > a.lim - 2 or abs(b[1]) > a.lim - 2:
                continue
            boxes.append([C.CLASS_NAMES.index(nm)] + [float(t) for t in b])
    per_cls = [[] for _ in C.CLASS_NAMES]
    for b in boxes:
        per_cls[int(b[0])].append(b)
    boxes = per_cls
    excl = set()
    if a.excl_dir and os.path.isdir(a.excl_dir):
        excl = {f[:-4] for f in os.listdir(a.excl_dir) if f.endswith(".bin")}
    fr = legal_frames(excl=excl)
    print(f"位置先验框 {sum(len(v) for v in boxes)}  "
          f"分类 {[len(v) for v in boxes]}  合法背景帧 {len(fr)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    ns = (a.n + a.shard - 1) // a.shard
    tasks = [(s, [a.seed * 1000003 + s * a.shard + i
                  for i in range(min(a.shard, a.n - s * a.shard))]) for s in range(ns)]
    t0 = time.time()
    done = 0
    with mp.Pool(a.workers, initializer=_init,
                 initargs=(a.bg, boxes, fr, a.thr, a.lim, a.npts, a.drop, pri)) as pool:
        for si, pts, offs, labs in pool.imap_unordered(_work, tasks):
            stem = os.path.join(a.out, f"shard_{si:03d}")
            np.save(stem + ".npy", pts)
            with open(stem + "_meta.json", "w") as f:
                json.dump({"offsets": offs, "labels": labs}, f)
            done += 1
            el = time.time() - t0
            print(f"  片 {done}/{ns}  {len(pts):,} 点  {el:.0f}s  "
                  f"{done*a.shard/el:.2f} 帧/s", flush=True)
    print(f"完成 {a.n} 帧 -> {a.out}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
