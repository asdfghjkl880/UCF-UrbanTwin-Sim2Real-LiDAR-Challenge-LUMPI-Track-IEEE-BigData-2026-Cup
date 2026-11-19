"""快速 AP（基于 pred_ada 的 npz 缓存）：距离预筛 + 逐帧贪心匹配。

官方 _iou_3d 每对都建 shapely 多边形，133 帧 × 7 类会跑到几十分钟。
这里先用「中心距 < 半对角和」筛掉不可能 IoU>=0.5 的配对，只对剩余候选
调用官方 IoU，结果一致但快一到两个数量级。
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
from detection_eval import _iou_3d  # noqa: E402
from ap1 import ap40  # noqa: E402

NAMES = C.CLASS_NAMES


def ap_one(z, ci, thr=0.0, tiou=0.5, topk=0):
    sc_all, tp_all = [], []
    n_gt = 0
    for fid in sorted(z):
        b, s, c, g = z[fid]
        gb = np.asarray(g, np.float64).reshape(-1, 8)
        gb = gb[gb[:, 0] == ci][:, 1:8]
        n_gt += len(gb)
        sel = np.flatnonzero((c == ci) & (s >= thr))
        if not len(sel):
            continue
        sel = sel[np.argsort(-s[sel])]
        if topk:
            sel = sel[:topk]
        used = np.zeros(len(gb), bool)
        rd = 0.5 * np.hypot(gb[:, 3], gb[:, 4]) if len(gb) else np.zeros(0)
        for i in sel:
            sc_all.append(float(s[i]))
            bi = b[i]
            ri = 0.5 * float(np.hypot(bi[3], bi[4]))
            if not len(gb):
                tp_all.append(0)
                continue
            d = np.hypot(gb[:, 0] - bi[0], gb[:, 1] - bi[1])
            cand = np.flatnonzero((~used) & (d < ri + rd + 0.01))
            hit = 0
            if len(cand):
                best, bj = 0.0, -1
                for j in cand:
                    v = _iou_3d(bi, gb[j])
                    if v > best:
                        best, bj = v, j
                if best >= tiou:
                    used[bj] = True
                    hit = 1
            tp_all.append(hit)
    n_pred = len(sc_all)
    if n_gt == 0 or n_pred == 0:
        return 0.0, n_gt, n_pred
    sc = np.asarray(sc_all)
    tp = np.asarray(tp_all, bool)
    return ap40(sc, tp, n_gt), n_gt, n_pred


def mean_ap(z, thr_map=None, tiou=0.5, topk_map=None, verbose=False):
    thr_map = thr_map or {}
    topk_map = topk_map or {}
    vals, per = [], {}
    for ci, nm in enumerate(NAMES):
        v, ng, npd = ap_one(z, ci, thr_map.get(nm, 0.0), tiou,
                            topk_map.get(nm, 0))
        per[nm] = (v, ng, npd)
        if ng > 0:
            vals.append(v)
        if verbose:
            print(f"  {nm:<11}{v:8.4f}{ng:7d}{npd:8d}")
    return float(np.mean(vals)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_sbg1_h.npz.npy"))
    ap.add_argument("--mode", default="base")
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    if not hasattr(z, "keys"):
        z = z.item()

    m, _ = mean_ap(z, verbose=True)
    print(f"mAP@0.5 = {m:.4f}")

    if a.mode == "van":
        print(f"\n{'tv':>6}{'topk':>7}{'mAP':>8}{'VanAP':>8}{'VanPred':>9}")
        best = (m, 0.0, 0)
        for tv in (0.0, 0.05, 0.10, 0.20, 0.35, 0.5, 0.7):
            for tk in (0, 5, 10, 20, 40):
                mm, per = mean_ap(z, {"Van": tv, "Bus": tv, "Truck": tv},
                                  topk_map={"Van": tk, "Bus": tk, "Truck": tk})
                print(f"{tv:>6.2f}{tk:>7}{mm:>8.4f}{per['Van'][0]:>8.4f}"
                      f"{per['Van'][2]:>9}")
                if mm > best[0]:
                    best = (mm, tv, tk)
        print(f"\n最佳 Van thr={best[1]} topk={best[2]}  mAP={best[0]:.4f}")


if __name__ == "__main__":
    main()
