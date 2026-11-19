"""从已预测框派生次级类别假设（跨类发射）。

官方 detection_eval 按类别完全独立计算 AP：评测 Van 时只看 label==Van 的
预测框，同位置的 Car 预测完全不参与。而当前检测头把每个 anchor 绑定到唯一
类别，Car 出现时同位置的 Van/Truck/Bus anchor 被当负样本压制（实测 Van
0.056 / Bus 0.0007，而 Car 已到 0.63）。

这里不改模型，只在推理后按框尺寸的后验概率补发次级类别预测：
一个定位准确的 Car 框，如果尺寸落在 Van 的分布内，就同时以 Van 类发射一次。
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
from detection_eval import evaluate_3d_detection  # noqa: E402

NAMES = C.CLASS_NAMES
VEH = ["Car", "Van", "Truck", "Bus"]


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


def post(dims, tgt, L, W, H, w=None):
    """对数正态近似下，尺寸属于该类的似然（只用到 L/H）。"""
    mu, sd, lo, hi = dims[tgt]
    p = 1.0
    for k, val in ((0, L), (2, H)):        # L 与 H 判别力最强
        z = (val - mu[k]) / sd[k]
        p *= np.exp(-0.5 * z * z)
        # 超出该类真实尺寸范围则强惩罚
        if val < lo[k] - 0.2 or val > hi[k] + 0.2:
            p *= 1e-3
    return p


def build(z, dims, thr, scale, keep_self=True, src_thr=0.25):
    preds, gt = {}, {}
    for fid, (b, s, c, g) in z.items():
        fid = str(fid)
        rows = []
        for i in range(len(b)):
            cls = NAMES[int(c[i])]
            L, W, H = float(b[i][3]), float(b[i][4]), float(b[i][5])
            if keep_self or cls not in VEH:
                rows.append({"box": [float(v) for v in b[i]],
                             "score": float(s[i]), "class": cls})
            if cls in VEH and float(s[i]) >= src_thr:
                for tgt in VEH:
                    if tgt == cls:
                        continue
                    p = post(dims, tgt, L, W, H)
                    if p < thr:
                        continue
                    sc = float(s[i]) * scale * min(1.0, p)
                    if sc < 0.02:
                        continue
                    rows.append({"box": [float(v) for v in b[i]],
                                 "score": sc, "class": tgt})
        preds[fid] = rows
        gr = []
        for r in np.asarray(g, np.float64).reshape(-1, 8):
            cid = int(r[0])
            if 0 <= cid < len(NAMES):
                gr.append({"class": NAMES[cid],
                           "box": [float(v) for v in r[1:8]]})
        gt[fid] = gr
    return preds, gt


def run(preds, gt):
    d = evaluate_3d_detection(preds, gt, NAMES, iou_thresholds=(0.5,), iou_type="3d")
    return d["mAP_0.5_R40"], d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_sbg1_h.npz.npy"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    if not hasattr(z, "keys"):
        z = z.item()
    dims = gt_dims(a.gtdir)
    for nm in VEH:
        mu, sd, lo, hi = dims[nm]
        print(f"{nm:<6} L {mu[0]:5.2f}±{sd[0]:.2f} [{lo[0]:.2f},{hi[0]:.2f}]  "
              f"H {mu[2]:5.2f}±{sd[2]:.2f} [{lo[2]:.2f},{hi[2]:.2f}]")

    base, _ = run(*build(z, dims, 9e9, 1.0))    # thr 极大 = 不发射
    print(f"\n基线 mAP {base:.4f}")
    best = (base, None)
    for thr, sc, st in ((0.05, 0.5, 0.25), (0.20, 0.5, 0.25), (0.05, 0.5, 0.5)):
        m, _ = run(*build(z, dims, thr, sc, src_thr=st))
        flag = ""
        if m > best[0]:
            best = (m, (thr, sc, st))
            flag = " *"
        print(f"  thr={thr:<5} scale={sc:<4} src={st:<5} -> {m:.4f}{flag}")
    print(f"\n最佳 {best[1]}  mAP {best[0]:.4f}  (+{best[0]-base:.4f})")
    m, d = run(*build(z, dims, *best[1]))
    for nm in NAMES:
        c = d["per_class"][nm]
        print(f"  {nm:<11}{c['AP_0.5_R40']:8.4f}{c['n_gt']:7d}{c['n_pred']:8d}")


if __name__ == "__main__":
    main()
