"""把两份 predictions.json（单次推理 / 镜像 TTA）按类混合，再叠加跨类发射。

holdout 实测 TTA 的收益是分化的：Person +0.018、Car +0.009 为正，
Bicycle -0.035、Moto -0.016 为负。所以提交包里只对 Person/Car 取 TTA 结果。

用法:
  py scripts/mix_json.py --plain _v18raw.zip --tta _v18tta.zip \
      --out submission_v19.zip --tta-classes Person,Car
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
from launch_extra import gt_dims, post, VEH  # noqa: E402
from rerank_apply import apply_rows_full  # noqa: E402
from cloud_io import frame_cloud, support_diag, check_support  # noqa: E402


def read_zip(p):
    z = zipfile.ZipFile(p)
    P = json.loads(z.read("predictions.json"))
    decl = z.read("declaration.txt").decode("utf-8") \
        if "declaration.txt" in z.namelist() else ""
    syn = sorted(x for x in z.namelist() if x.startswith("synthetic/"))
    return P, decl, [(nm, z.read(nm)) for nm in syn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--tta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gtdir", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--tta-classes", default="Person,Car")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--src-thr", type=float, default=0.1)
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--rerank-cfg",
                    default=os.path.join(ROOT, "reports/rerank_pts.json"))
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--center-cfg",
                    default=os.path.join(ROOT, "reports/center_fix.json"))
    ap.add_argument("--no-center", action="store_true")
    ap.add_argument("--yaw-cfg", default=os.path.join(ROOT, "reports/yaw_fix.json"))
    a = ap.parse_args()

    P, decl, syn = read_zip(a.plain)
    T, _, _ = read_zip(a.tta)
    keep = {x.strip() for x in a.tta_classes.split(",") if x.strip()}
    print(f"TTA 类 {sorted(keep)}   帧数 plain {len(P)} / tta {len(T)}")

    dims = gt_dims(a.gtdir)
    n_mix = n_launch = n_plain = 0
    Q = {}
    for fid, rows in P.items():
        cur = [r for r in rows if r["class"] not in keep]
        n_plain += len(cur)
        trows = [r for r in T.get(fid, []) if r["class"] in keep]
        n_mix += len(trows)
        cur += trows

        if not a.no_launch:
            add = []
            for r in cur:
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
                    add.append({"box": list(r["box"]), "score": sc, "class": tgt})
            n_launch += len(add)
            cur += add

        cur.sort(key=lambda x: -x["score"])
        Q[fid] = cur

    print(f"单次 {n_plain}  TTA {n_mix}  发射 {n_launch}  "
          f"合计 {n_plain+n_mix+n_launch}  平均 {(n_plain+n_mix+n_launch)/max(len(Q),1):.0f} 框/帧")

    # 点云实测特征重排序：必须在**该帧预测所在的真实输入点云**上统计。
    # 曾经这里遍历的是 syn（随包提交的合成帧，供 realism 用），与检测帧无关：
    # 真目标框内点数为 0 -> 分数被乘到 0.01 保底 -> 线上 Person 0.2666→0.0034。
    if not a.no_rerank and os.path.exists(a.rerank_cfg):
        cfg = json.load(open(a.rerank_cfg))
        tmap, ymap = {}, {}
        if not a.no_center and os.path.exists(a.center_cfg):
            tmap = json.load(open(a.center_cfg))
        if not a.no_center and os.path.exists(a.yaw_cfg):
            ymap = json.load(open(a.yaw_cfg))
        n_row = n_cen = 0
        acc = {}
        for fid in sorted(Q):
            pts = frame_cloud(fid)
            support_diag(Q[fid], pts, acc)
            x, y = apply_rows_full(Q[fid], pts, cfg, tmap, ymap)
            n_row += x
            n_cen += y
        check_support(acc)
        # 重排序后的权重可以 >1；用一个全局单调映射 s->s/(1+s) 压回 [0,1)。
        # 单调变换不改变任何一类的排序，因此 AP 完全不变（评测只按 score 排序，无阈值）。
        mx = max((r["score"] for v in Q.values() for r in v), default=1.0)
        if mx > 1.0:
            for v in Q.values():
                for r in v:
                    r["score"] = float(r["score"]) / (1.0 + float(r["score"]))
        Q = {f: sorted(v, key=lambda x: -x["score"]) for f, v in Q.items()}
        print(f"重排序 {n_row} 行 / 中心修正 {n_cen} 行")
        print(f"  权重 {cfg}")
        print(f"  中心修正 t { {k: v for k, v in tmap.items() if v} }")
        print(f"  朝向修正 t { {k: v for k, v in ymap.items() if v} }")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as o:
        for k, (nm, data) in enumerate(syn):
            o.writestr(f"synthetic/{k+1:04d}.bin", data)
        o.writestr("predictions.json", json.dumps(Q))
        if decl:
            o.writestr("declaration.txt", decl)
    print(f"-> {a.out}  ({len(syn)} 帧, {os.path.getsize(a.out)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
