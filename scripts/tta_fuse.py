"""TTA 共识融合：把多视角预测按 BEV 一致性聚成簇，用"支持数"当置信度。

原来的 detect_tta 把多视角结果直接 NMS、只留最高分，一致性的证据被丢弃。
这里改为：
  * 逐类把 4 个视角的框按 BEV IoU 聚类（取分数最高的作为簇代表）；
  * score = (a·max + (1-a)·mean) · (support/NP)^gamma
    —— 只有多视角都看到的框才能保持高分；
  * 中心与朝向取簇内加权平均（朝向用 cos2θ/sin2θ 平均，规避 π 歧义）；
  * 尺寸保留簇代表的（平均尺寸会缩小框，反而降 IoU）。

用法:
  py scripts/tta_fuse.py --passes data/out/_p4h.npz.npy --out data/out/_f4h.npz
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pp_config as C  # noqa: E402
from pp_model import bev_iou_matrix  # noqa: E402

NPK = ("p0", "p1", "p2", "p3")


def fuse_frame(rec, tau=0.45, alpha=0.5, gamma=1.0, avg_dims=False):
    """rec: {pass_name: (b, s, c)} -> (B, S, K)。"""
    B, S, K, P = [], [], [], []
    for pi, nm in enumerate(NPK):
        if nm not in rec:
            continue
        b, s, c = rec[nm]
        if not len(b):
            continue
        B.append(np.asarray(b, np.float64))
        S.append(np.asarray(s, np.float64).ravel())
        K.append(np.asarray(c).astype(np.int64))
        P.append(np.full(len(b), pi, np.int64))
    if not B:
        return (np.zeros((0, 7), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.int64))
    B, S, K, P = (np.vstack(B), np.concatenate(S),
                  np.concatenate(K), np.concatenate(P))
    npass = len(set(P.tolist()))

    CELL = 6.0
    ob, os_, ok = [], [], []
    for ci in np.unique(K):
        m = np.flatnonzero(K == ci)
        bb, ss, pp = B[m], S[m], P[m]
        order = np.argsort(-ss)
        clusters = {}            # 代表下标 -> 成员下标列表
        grid = {}                # (ix, iy) -> [代表下标, ...]
        for i in order:
            i = int(i)
            gx = int(np.floor(bb[i, 0] / CELL))
            gy = int(np.floor(bb[i, 1] / CELL))
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    e = grid.get((gx + dx, gy + dy))
                    if e:
                        cand.extend(e)
            bj = -1
            if cand:
                ca = np.asarray(sorted(set(cand)), np.int64)
                d = np.hypot(bb[ca, 0] - bb[i, 0], bb[ca, 1] - bb[i, 1])
                half = (0.5 * np.hypot(bb[ca, 3], bb[ca, 4])
                        + 0.5 * np.hypot(bb[i, 3], bb[i, 4]))
                keep = d < half * 1.1
                if keep.any():
                    ca, d = ca[keep], d[keep]
                    # 只对最近 3 个簇代表做精确 IoU（shapely 调用是大头）
                    if len(ca) > 3:
                        ca = ca[np.argsort(d)[:3]]
                    iou = bev_iou_matrix(bb[i:i + 1], bb[ca])[0]
                    j = int(np.argmax(iou))
                    if iou[j] > tau:
                        bj = int(ca[j])
            if bj >= 0:
                clusters[bj].append(i)
            else:
                clusters[i] = [i]
                grid.setdefault((gx, gy), []).append(i)
        for cl in clusters.values():
            idx = np.asarray(cl)
            w = ss[idx] + 1e-6
            w = w / w.sum()
            sup = len(set(pp[idx].tolist()))
            sc = (alpha * ss[idx].max() + (1 - alpha) * (ss[idx] * w).sum())
            sc *= (sup / npass) ** gamma
            box = bb[cl[0]].copy()
            box[0] = float((bb[idx, 0] * w).sum())
            box[1] = float((bb[idx, 1] * w).sum())
            box[2] = float((bb[idx, 2] * w).sum())
            if avg_dims:
                box[3] = float((bb[idx, 3] * w).sum())
                box[4] = float((bb[idx, 4] * w).sum())
                box[5] = float((bb[idx, 5] * w).sum())
            c2 = float((np.cos(2 * bb[idx, 6]) * w).sum())
            s2 = float((np.sin(2 * bb[idx, 6]) * w).sum())
            box[6] = 0.5 * float(np.arctan2(s2, c2))
            ob.append(box)
            os_.append(sc)
            ok.append(int(ci))
    return (np.asarray(ob, np.float32), np.asarray(os_, np.float32),
            np.asarray(ok, np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", default=os.path.join(ROOT, "data/out/_p4h.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/out/_f4h.npz"))
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--avg-dims", type=int, default=0)
    ap.add_argument("--pick", default="",
                    help="tta_pick.json：只对指定类用融合结果，其余类保留 p0")
    a = ap.parse_args()
    p = a.passes if a.passes.endswith(".npy") else a.passes + ".npy"
    z = np.load(p, allow_pickle=True)
    z = z if hasattr(z, "keys") else z.item()
    pick = {}
    if a.pick and os.path.exists(a.pick):
        pick = json.load(open(a.pick))
        print("逐类选择: " + str({k: v for k, v in pick.items() if v == "fused"}))

    out = {}
    for fid, (rec, gt) in z.items():
        b, s, c = fuse_frame(rec, a.tau, a.alpha, a.gamma, bool(a.avg_dims))
        if pick:
            b0, s0, c0 = rec["p0"]
            c0 = np.asarray(c0).astype(np.int64)
            keep = np.array([i for i, nm in enumerate(C.CLASS_NAMES)
                             if pick.get(nm) == "fused"], np.int64)
            # 融合结果只保留「选中的类」；其余类一律用原始视角的框
            cf = np.asarray(c, np.int64)
            mf = np.isin(cf, keep) if len(keep) else np.zeros(len(cf), bool)
            m0 = ~np.isin(c0, keep) if len(keep) else np.ones(len(c0), bool)
            b = np.vstack([np.asarray(b, np.float32)[mf],
                           np.asarray(b0, np.float32)[m0]])
            s = np.concatenate([np.asarray(s, np.float32).ravel()[mf],
                                np.asarray(s0, np.float32).ravel()[m0]])
            c = np.concatenate([cf[mf], c0[m0]])
        out[fid] = (b.astype(np.float32), s.astype(np.float32),
                    c.astype(np.int64), gt)
    np.save(a.out, out, allow_pickle=True)
    n = sum(len(v[0]) for v in out.values())
    print(f"写出 {a.out}  {len(out)} 帧  {n} 框 ({n/max(len(out),1):.0f}/帧)")


if __name__ == "__main__":
    main()
