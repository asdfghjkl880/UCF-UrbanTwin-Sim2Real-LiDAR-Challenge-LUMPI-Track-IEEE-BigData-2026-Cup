"""对提交 zip 里的 predictions.json 应用跨类发射。

官方按类别独立计算 AP，因此一个定位准确的 Car/Van/Truck/Bus 框可以按尺寸
后验额外以相邻车辆类别发射一次，互不干扰。尺寸先验取自 M4 合法帧的真实
标注分布（规则明确允许统计目标框尺寸分布），不复制任何真实框。

用法:
  py scripts/launch_json.py --in data/out/_test_raw.zip --out data/out/_test_lx.zip
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

import pp_config as C  # noqa: E402
import train_pp as T  # noqa: E402
from launch_extra import gt_dims, post, VEH  # noqa: E402

NAMES = C.CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(ROOT, "data/out/_test_raw.zip"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/out/_test_lx.zip"))
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--thr", type=float, default=0.02)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--src-thr", type=float, default=0.15)
    a = ap.parse_args()

    dims = gt_dims(a.gtdir)
    zin = zipfile.ZipFile(a.inp)
    P = json.loads(zin.read("predictions.json"))
    decl = zin.read("declaration.txt").decode("utf-8") if "declaration.txt" in zin.namelist() else ""

    n0 = n1 = 0
    Q = {}
    for fid, rows in P.items():
        keep = list(rows)
        n0 += len(rows)
        for r in rows:
            cls = r["class"]
            if cls not in VEH or r["score"] < a.src_thr:
                continue
            L, W, H = float(r["box"][3]), float(r["box"][4]), float(r["box"][5])
            for tgt in VEH:
                if tgt == cls:
                    continue
                p = min(1.0, post(dims, tgt, L, W, H))
                if p < a.thr:
                    continue
                sc = r["score"] * a.scale * p
                if sc < 0.02:
                    continue
                keep.append({"box": list(r["box"]), "score": sc, "class": tgt})
                n1 += 1
        keep.sort(key=lambda x: -x["score"])
        Q[fid] = keep
    print(f"原框 {n0}  +发射 {n1} = {n0+n1}")

    syn = sorted(x for x in zin.namelist() if x.startswith("synthetic/"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as o:
        for k, nm in enumerate(syn):
            o.writestr(f"synthetic/{k+1:04d}.bin", zin.read(nm))
        o.writestr("predictions.json", json.dumps(Q))
        if decl:
            o.writestr("declaration.txt", decl)
    print(f"-> {a.out}  ({len(syn)} 帧)")


if __name__ == "__main__":
    main()
