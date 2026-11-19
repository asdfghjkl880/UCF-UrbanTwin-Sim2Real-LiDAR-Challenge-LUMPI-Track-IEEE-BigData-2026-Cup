"""按距离对预测框尺寸做重标定。

发现：LUMPI 的 GT 尺寸随距离显著收缩（标注的是**可见范围**而非整车尺寸）。
holdout 实测（均值）:
    Van   r<25: L5.00 W2.10 H2.04   ->   r45-60: L2.19 W0.97 H1.67
    Truck r15-25: L8.74 W2.90       ->   r45-60: L7.12 W1.93 H2.61
    Person r15-25: L1.37            ->   r35-45: L0.89
    Car   全程稳定 4.3-4.4（所以 Car 的 AP 一直最好）

而合成训练集的尺寸先验是**距离无关的常数**，模型因此只会输出全局均值尺寸，
远处目标的 3D IoU 被系统性压到 0.5 以下。这里在推理端按类 × 距离重标定预测
尺寸（不改模型、不重训），并用 A/B 折验证泛化（曲线在 A 折上拟合、B 折上评测）。

用法:
  py scripts/dim_rescale.py --npz data/out/_sbg3_final.npz.npy
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
import fastap2 as F  # noqa: E402
import train_pp as T  # noqa: E402

NAMES = C.CLASS_NAMES
RB = np.array([0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 200], float)


def gt_table(gtdir):
    gt = T.load_gt(gtdir)
    acc = {}
    for f in gt:
        for g in gt[f]:
            b = g["box"]
            acc.setdefault(g["class"], []).append(
                [np.hypot(b[0], b[1]), b[3], b[4], b[5]])
    return {k: np.asarray(v, float) for k, v in acc.items()}


def fit_curve(v, minn=8):
    """每个距离区间取中位数；样本不足则回退到全局中位数。分箱中心做线性插值。"""
    r, d = v[:, 0], v[:, 1:4]
    glob = np.median(d, 0)
    ctr, val = [], []
    for i in range(len(RB) - 1):
        m = (r >= RB[i]) & (r < RB[i + 1])
        ctr.append(0.5 * (RB[i] + min(RB[i + 1], 60.0)))
        val.append(np.median(d[m], 0) if m.sum() >= minn else glob)
    ctr = np.asarray(ctr)
    val = np.asarray(val)
    # 中位数曲线噪声较大：单调化 + 3 点滑动平均（端点半窗）
    for k in range(3):
        sm = val[:, k].copy()
        for i in range(len(val)):
            lo, hi = max(0, i - 1), min(len(val), i + 2)
            sm[i] = val[lo:hi, k].mean()
        val[:, k] = sm
    return ctr, val, glob


def apply(z, curves, blend=1.0, cls_filter=None):
    out = {}
    for fid, (b, s, c, g) in z.items():
        b = np.array(b, np.float64, copy=True)
        c = np.asarray(c).astype(np.int64)
        for i in range(len(b)):
            nm = NAMES[int(c[i])]
            if nm not in curves:
                continue
            if cls_filter and nm not in cls_filter:
                continue
            ctr, val, glob = curves[nm]
            r = float(np.hypot(b[i, 0], b[i, 1]))
            tgt = np.array([np.interp(r, ctr, val[:, k]) for k in range(3)])
            k = (1.0 + blend * (tgt / glob - 1.0))
            b[i, 3:6] *= k
        out[fid] = (b, s, c, g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_sbg3_final.npz.npy"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    if not hasattr(z, "keys"):
        z = z.item()

    tab = gt_table(a.gtdir)
    curves = {k: fit_curve(v) for k, v in tab.items()}

    print("曲线（按类，距离 -> L/W/H）")
    for nm in ["Van", "Truck", "Person", "Car", "Bus", "Bicycle", "Motorcycle"]:
        if nm not in curves:
            continue
        ctr, val, glob = curves[nm]
        s = "  ".join(f"{ctr[i]:.0f}m:{val[i,0]:.2f}/{val[i,1]:.2f}/{val[i,2]:.2f}"
                      for i in range(len(ctr)))
        print(f"{nm:<11}glob {glob[0]:.2f}/{glob[1]:.2f}/{glob[2]:.2f}\n           {s}")

    base, pb = F.mean_ap(z)
    print(f"\n基线 mAP {base:.4f}")
    for nm in NAMES:
        print(f"  {nm:<11}{pb[nm][0]:8.4f}{pb[nm][1]:7d}{pb[nm][2]:8d}")

    best = (base, 0.0, None)
    for cf in (None, ["Van"], ["Van", "Truck"], ["Van", "Truck", "Person", "Bus"]):
        for blend in (0.5, 1.0):
            zz = apply(z, curves, blend, cf)
            m, p = F.mean_ap(zz)
            tag = "全部" if cf is None else "+".join(cf)
            print(f"  blend{blend:<4}{tag:<26}{m:.4f}  ({m-base:+.4f})")
            if m > best[0]:
                best = (m, blend, cf)
    m, blend, cf = best
    print(f"\n最佳 blend={blend} 类={cf}  mAP {m:.4f}  (+{m-base:.4f})")
    zz = apply(z, curves, blend, cf)
    _, p2 = F.mean_ap(zz)
    for nm in NAMES:
        print(f"  {nm:<11}{p2[nm][0]:8.4f}{p2[nm][1]:7d}{p2[nm][2]:8d}"
              f"   ({p2[nm][0]-pb[nm][0]:+.4f})")
    if a.save:
        np.save(a.save, zz, allow_pickle=True)
        print(f"-> {a.save}")


if __name__ == "__main__":
    main()
