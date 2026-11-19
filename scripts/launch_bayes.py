"""贝叶斯归一化的跨类发射（替代裸似然 × 全局 scale）。

原方案 sc = s * scale * p_tgt，其中 p_tgt 是"尺寸属于该类"的裸似然。
问题：Van 与 Car 的尺寸分布大量重叠，一个典型 Car 对 Van 的裸似然也有
0.3~0.5，于是海量真 Car 发射出的 Van 假框把少数真 Van 压在排序后面
（实测 36,767 个 Van 预测里只有约 30 个 TP）。

改成类间竞争：对同一框计算 Car/Van/Truck/Bus 四类的 log 似然并 softmax，
得到"在四个候选里它属于 Van 的概率"。典型 Car 的这个后验会接近 0，
而典型 Van 会接近 1，排序因此被拉开。再乘 gamma 次方进一步锐化。

用法:
  py scripts/launch_bayes.py --npz data/out/_sbg3_h.npz.npy
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
import train_pp as T  # noqa: E402
import fastap2 as F  # noqa: E402

NAMES = C.CLASS_NAMES
VEH = ["Car", "Van", "Truck", "Bus"]
VIDX = [NAMES.index(v) for v in VEH]


def gt_dims(gtdir):
    gt = T.load_gt(gtdir)
    acc = {}
    for f in gt:
        for g in gt[f]:
            nm = g.get("class")
            if nm in NAMES:
                acc.setdefault(nm, []).append(g["box"][3:6])
    out = {}
    for nm, v in acc.items():
        v = np.asarray(v, np.float64)
        out[nm] = (v.mean(0), v.std(0) + 1e-3, v.min(0), v.max(0))
    return out


def loglik(dims, L, W, H, useW=True):
    """四类 log 似然（-0.5 z^2 之和，越界加 -7）。"""
    lp = np.zeros(len(VEH), np.float64)
    for i, nm in enumerate(VEH):
        mu, sd, lo, hi = dims[nm]
        ks = (0, 1, 2) if useW else (0, 2)
        for k in ks:
            val = (L, W, H)[k]
            z = (val - mu[k]) / sd[k]
            lp[i] += -0.5 * z * z
            if val < lo[k] - 0.2 or val > hi[k] + 0.2:
                lp[i] += -7.0
    return lp


def softmax(lp):
    m = lp.max()
    e = np.exp(lp - m)
    return e / e.sum()


def build(z, dims, gamma=1.0, scale=1.0, src_thr=0.15, useW=True,
          launch_floor=0.0, drop_src_veh=False):
    """在 z 上追加发射框，返回新的 z（同格式，boxes/scores/classes/gt）。

    drop_src_veh=True 时把原车辆类预测中"后验不像自己"的丢掉（纯化）。
    """
    out = {}
    for fid, (b, s, c, g) in z.items():
        b = np.asarray(b, np.float64)
        s = np.asarray(s, np.float64).ravel()
        c = np.asarray(c).astype(np.int64).ravel()
        cls_ok = np.isin(c, VIDX)
        # 计算所有"源 = 车辆类"框的后验
        post = np.zeros((len(b), len(VEH)), np.float64)
        for i in np.flatnonzero(cls_ok):
            post[i] = softmax(loglik(dims, b[i, 3], b[i, 4], b[i, 5], useW))

        keep_b, keep_s, keep_c = [b], [s], [c]
        src = np.flatnonzero(cls_ok & (s >= src_thr))
        nb, ns, nc = [], [], []
        for i in src:
            src_i = VIDX.index(int(c[i]))
            for j, tgt in enumerate(VIDX):
                if j == src_i:
                    continue
                p = post[i, j] ** gamma
                if p < launch_floor:
                    continue
                sc = s[i] * scale * p
                if sc < 0.02:
                    continue
                nb.append(b[i])
                ns.append(sc)
                nc.append(tgt)
        if nb:
            keep_b.append(np.asarray(nb, np.float64))
            keep_s.append(np.asarray(ns, np.float64))
            keep_c.append(np.asarray(nc, np.int64))

        if drop_src_veh:
            # 只保留非车辆类 + 车辆类中后验 >= 0.3 的
            own = np.zeros(len(b), np.float64)
            for i in np.flatnonzero(cls_ok):
                own[i] = post[i, VIDX.index(int(c[i]))]
            bad = cls_ok & (own < 0.3)
            sel = ~bad
            keep_b = [keep_b[0][sel]]
            keep_s = [keep_s[0][sel]]
            keep_c = [keep_c[0][sel]]

        out[fid] = (np.vstack(keep_b), np.concatenate(keep_s),
                    np.concatenate(keep_c), g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_sbg3_h.npz.npy"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    if not hasattr(z, "keys"):
        z = z.item()
    dims = gt_dims(a.gtdir)
    for nm in VEH:
        mu, sd, lo, hi = dims[nm]
        print(f"{nm:<6} L {mu[0]:5.2f}±{sd[0]:.2f}  W {mu[1]:5.2f}±{sd[1]:.2f}  "
              f"H {mu[2]:5.2f}±{sd[2]:.2f}   n_gt 范围 L[{lo[0]:.2f},{hi[0]:.2f}] "
              f"H[{lo[2]:.2f},{hi[2]:.2f}]")

    base, per = F.mean_ap(z, verbose=True)
    print(f"\n基线 mAP {base:.4f}\n")

    if not a.sweep:
        zz = build(z, dims, gamma=2.0, scale=1.0, src_thr=0.15)
        m, per2 = F.mean_ap(zz, verbose=True)
        print(f"\nmAP {m:.4f}  ({m-base:+.4f})")
        return

    best = (base, None)
    for useW in (True, False):
        for gamma in (1.0, 2.0, 3.0, 4.0):
            for scale in (0.6, 1.0):
                for st in (0.10, 0.15):
                    zz = build(z, dims, gamma=gamma, scale=scale,
                               src_thr=st, useW=useW)
                    m, p2 = F.mean_ap(zz)
                    tag = ""
                    if m > best[0]:
                        best = (m, (useW, gamma, scale, st))
                        tag = " *"
                    print(f"  W={int(useW)} g={gamma:<4} sc={scale:<4} st={st:<5}"
                          f" -> {m:.4f}  Van {p2['Van'][0]:.4f}{tag}", flush=True)
    print(f"\n最佳 {best[1]}  mAP {best[0]:.4f}  ({best[0]-base:+.4f})")
    useW, gamma, scale, st = best[1]
    m, p2 = F.mean_ap(build(z, dims, gamma=gamma, scale=scale, src_thr=st,
                            useW=useW), verbose=True)


if __name__ == "__main__":
    main()
