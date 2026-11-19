#!/usr/bin/env python3
"""用 TTA 共识融合的预测重建提交包（点云与声明沿用既有包，不重跑合成）。

流程：融合后的 npz -> 跨类发射 -> 点云重排序/中心与朝向修正 -> 分数压缩 -> zip
用法:
    py scripts/pack_fused.py --base data/out/submission_v22.zip \
        --fused data/out/_f4test.npz --out data/out/submission_v23.zip
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
from launch_extra import gt_dims  # noqa: E402
from fuse_final import launch  # noqa: E402
from rerank_apply import apply_rows_full  # noqa: E402
from cloud_io import frame_cloud, support_diag, check_support  # noqa: E402


def load(p):
    z = np.load(p, allow_pickle=True)
    return z if hasattr(z, "keys") else z.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "data/out/submission_v22.zip"))
    ap.add_argument("--fused", default=os.path.join(ROOT, "data/out/_f4test.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/out/submission_v23.zip"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--clouds", default=os.path.join(ROOT, "data/eval_m4_test"))
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--rerank-cfg",
                    default=os.path.join(ROOT, "reports/rerank_pts.json"))
    ap.add_argument("--center-cfg",
                    default=os.path.join(ROOT, "reports/center_fix.json"))
    ap.add_argument("--yaw-cfg", default=os.path.join(ROOT, "reports/yaw_fix.json"))
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()

    f = a.fused if a.fused.endswith(".npy") else a.fused + ".npy"
    F = load(f)
    n0 = sum(len(v[0]) for v in F.values())
    print(f"融合输入 {len(F)} 帧 {n0} 框 ({n0/max(len(F),1):.0f}/帧)")

    if not a.no_launch:
        dims = gt_dims(a.gtdir)
        F = launch(F, dims, scale=a.scale)
    n1 = sum(len(v[0]) for v in F.values())
    print(f"发射后 {n1} 框")

    cfg = json.load(open(a.rerank_cfg)) if os.path.exists(a.rerank_cfg) else {}
    tmap = json.load(open(a.center_cfg)) if os.path.exists(a.center_cfg) else {}
    ymap = json.load(open(a.yaw_cfg)) if os.path.exists(a.yaw_cfg) else {}

    with zipfile.ZipFile(a.base) as zf:
        names = zf.namelist()
        syn = [(n, zf.read(n)) for n in sorted(
            [x for x in names if x.startswith("synthetic/")])]
        decl = zf.read("declaration.txt").decode("utf-8") if "declaration.txt" in names else ""

    ids = sorted(F)
    Q, n_row = {}, 0
    acc = {}
    for fid in ids:
        b, s, c, _g = F[fid]
        b = np.asarray(b, np.float64)
        s = np.asarray(s, np.float64).ravel()
        c = np.asarray(c).astype(np.int64).ravel()
        rows = [{"box": [float(v) for v in b[i]], "score": float(s[i]),
                 "class": C.CLASS_NAMES[int(c[i])]} for i in range(len(b))]
        # 必须是这一帧的真实输入点云；合成帧 synthetic/*.bin 只用于 realism。
        pts = frame_cloud(fid)
        if not a.no_rerank and cfg:
            support_diag(rows, pts, acc)
            apply_rows_full(rows, pts, cfg, tmap, ymap)
            n_row += len(rows)
        Q[fid] = sorted(rows, key=lambda x: -x["score"])
    if not a.no_rerank and cfg:
        check_support(acc)
    print(f"重排序/几何修正 {n_row} 行")

    mx = max((r["score"] for v in Q.values() for r in v), default=1.0)
    if mx > 1.0:
        for v in Q.values():
            for r in v:
                r["score"] = float(r["score"]) / (1.0 + float(r["score"]))

    ntot = sum(len(v) for v in Q.values())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as o:
        for k, (nm, data) in enumerate(syn):
            o.writestr(f"synthetic/{k+1:04d}.bin", data)
        o.writestr("predictions.json", json.dumps(Q))
        if decl:
            o.writestr("declaration.txt", decl)
    print(f"-> {a.out}  {ntot} 框 ({ntot/max(len(Q),1):.0f}/帧)  "
          f"{os.path.getsize(a.out)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
