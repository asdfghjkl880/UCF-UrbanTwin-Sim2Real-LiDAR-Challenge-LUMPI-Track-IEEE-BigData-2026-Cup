#!/usr/bin/env python3
"""用官方 scoring_program/detection_eval 给本地预测打分。

与 bench_geo.py 的旧口径并列输出，用来量化两者差异。两处关键区别：

  1. 类的平均范围
     官方：所有 n_gt>0 的类都进平均（某类 n_gt=0 才算 NaN 排除）
     旧口径：只把"有预测结果"的类放进分母，完全没有预测的类被悄悄剔除

  2. 某帧缺该类 GT 时
     官方：该帧同类预测仍计入全局 FP（detection_eval.py 有明确注释）
     旧口径：直接 continue，这些 FP 不计

两条都会让旧口径偏高，所以本脚本以官方为准，旧口径仅作对照。

用法:
    py scripts/eval_official.py --npz data/out/_probe40.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
from detection_eval import evaluate_3d_detection, _iou_3d  # noqa: E402
from ap1 import ap40  # noqa: E402

NAMES = C.CLASS_NAMES


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    if not hasattr(z, "keys"):
        z = z.item()
    return z


def to_official(z):
    """转成官方需要的 {fid: [{'box','score','class'}]} 与 GT 同构格式。"""
    preds, gt = {}, {}
    for fid, (b, s, c, g) in z.items():
        fid = str(fid)
        preds[fid] = [
            {"box": [float(v) for v in b[i]], "score": float(s[i]),
             "class": NAMES[int(c[i])]}
            for i in range(len(b))
        ]
        rows = []
        for r in np.asarray(g, np.float64).reshape(-1, 8):
            cid = int(r[0])
            if not 0 <= cid < len(NAMES):
                continue
            rows.append({"class": NAMES[cid],
                         "box": [float(v) for v in r[1:8]]})
        gt[fid] = rows
    return preds, gt


def old_style(preds, gt):
    """复现 bench_geo 的口径，只用于量化差异，不作为决策依据。"""
    per = {}
    for nm in NAMES:
        S, T, ng = [], [], 0
        for fid, g in gt.items():
            gi = [x for x in g if x["class"] == nm]
            if not gi:
                continue  # <- 差异点 1：这些帧的同类 FP 不计
            ng += len(gi)
            pi = [p for p in preds.get(fid, []) if p["class"] == nm]
            if not pi:
                continue
            gb = np.asarray([x["box"] for x in gi], np.float64)
            pb = np.asarray([p["box"] for p in pi], np.float64)
            sc = np.asarray([p["score"] for p in pi], np.float64)
            o = np.argsort(-sc)
            pb, sc = pb[o], sc[o]
            used = np.zeros(len(gb), bool)
            tp = np.zeros(len(pb), bool)
            for i in range(len(pb)):
                best, bj = 0.0, -1
                for j in range(len(gb)):
                    if used[j]:
                        continue
                    v = _iou_3d(pb[i], gb[j])
                    if v > best:
                        best, bj = v, j
                if best >= 0.5 and bj >= 0:
                    used[bj] = True
                    tp[i] = True
            S.append(sc)
            T.append(tp)
        if ng == 0:
            per[nm] = float("nan")
            continue
        per[nm] = ap40(np.concatenate(S), np.concatenate(T), ng) if S else 0.0
    # <- 差异点 2：只取有结果的类
    live = [n for n in NAMES if per[n] == per[n] and per[n] > 0]
    return per, (float(np.mean([per[n] for n in live])) if live else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="pred_ada.py 输出的缓存")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    z = load_npz(a.npz)
    preds, gt = to_official(z)
    d5 = evaluate_3d_detection(preds, gt, NAMES, iou_thresholds=(0.5,),
                               iou_type="3d")
    d7 = evaluate_3d_detection(preds, gt, NAMES, iou_thresholds=(0.7,),
                               iou_type="3d")

    if not a.quiet:
        print(f"帧数 {len(gt)}   框 {d5['n_pred']}   GT {d5['n_gt']}   "
              f"框/帧 {d5['n_pred'] / max(len(gt), 1):.0f}")
        print(f"{'class':12s} {'AP@.5':>7s} {'AP@.7':>7s} {'n_gt':>6s} {'n_pred':>7s}")
        for nm in NAMES:
            c, c7 = d5["per_class"][nm], d7["per_class"][nm]
            print(f"{nm:12s} {c['AP_0.5_R40']:7.4f} {c7['AP_0.7_R40']:7.4f} "
                  f"{c['n_gt']:6d} {c['n_pred']:7d}")
    print(f"官方口径  mAP@0.5 = {d5['mAP_0.5_R40']:.4f}   "
          f"mAP@0.7 = {d7['mAP_0.7_R40']:.4f}")
    per_old, m_old = old_style(preds, gt)
    print(f"旧口径    mAP@0.5 = {m_old:.4f}   "
          f"倍率 = {m_old / max(d5['mAP_0.5_R40'], 1e-9):.2f}x")


if __name__ == "__main__":
    main()
