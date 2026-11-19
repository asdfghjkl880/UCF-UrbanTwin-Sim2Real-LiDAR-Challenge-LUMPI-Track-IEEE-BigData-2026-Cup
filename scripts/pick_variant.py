"""在**最终管线输出**上做逐类择优：融合变体 vs v22（TTA 按类混合）变体。

rerank / 中心修正 / 朝向修正 / 跨类发射都是逐框独立的，所以 AP 可以按帧子集
单独计算 —— 于是单元是「类 × 折」，只在该类两折都更优时才切换变体。
输出 reports/variant_pick.json: {类别: "fused" | "v22"}
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
import fastap2 as F  # noqa: E402


def load(p):
    z = np.load(p, allow_pickle=True)
    return z if hasattr(z, "keys") else z.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/out/_f4h_mix.npz.npy",
                    help="融合变体（最终管线输出前）")
    ap.add_argument("--b", default="data/out/_sbg3_v22.npz.npy",
                    help="v22 变体（TTA 按类混合）")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports/variant_pick.json"))
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    fids = sorted(set(A) & set(B))
    fa = set(fids[::2])
    pa = {f: A[f] for f in fids if f in fa}
    pb = {f: B[f] for f in fids if f in fa}
    qa = {f: A[f] for f in fids if f not in fa}
    qb = {f: B[f] for f in fids if f not in fa}

    pick, best = {}, 0.0
    print(f"{'class':11s}{'A_融合':>9}{'A_v22':>9}{'':>2}{'B_融合':>9}{'B_v22':>9}{'':>2} 决定")
    for nm in C.CLASS_NAMES:
        ci = C.CLASS_NAMES.index(nm)
        a1, a0 = F.ap_one(pa, ci)[0], F.ap_one(pb, ci)[0]
        b1, b0 = F.ap_one(qa, ci)[0], F.ap_one(qb, ci)[0]
        use_fused = (a1 > a0) and (b1 > b0)
        pick[nm] = "fused" if use_fused else "v22"
        best += (a1 + b1) / 2 if use_fused else (a0 + b0) / 2
        print(f"{nm:11s}{a1:9.4f}{a0:9.4f}{'':>2}{b1:9.4f}{b0:9.4f}{'':>2} "
              f"{'融合' if use_fused else 'v22 '}  "
              f"({a1-a0:+.4f}/{b1-b0:+.4f})")
    print(f"\n两折平均 mAP = {best/len(C.CLASS_NAMES):.4f}")
    json.dump(pick, open(a.out, "w"), indent=1)
    print(f"-> {a.out}  采用融合: {[k for k, v in pick.items() if v == 'fused']}")


if __name__ == "__main__":
    main()
