#!/usr/bin/env python3
"""按类合并两个提交变体的预测（逐类选择，不混合同类内的框）。

AP 是按类独立计算的，且每个类只取自一个变体，因此同类内的分数排序完全
保持原样；跨类分数不可比也没关系（评测从不在类间比较）。

用法:
    py scripts/merge_subs.py --a data/out/submission_v23a.zip \
        --b data/out/submission_v22.zip --pick reports/variant_pick.json \
        --out data/out/submission_v23.zip
"""
import argparse
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(ROOT, "data/out/submission_v23a.zip"),
                    help="融合变体（pick 值为 fused 时取它）")
    ap.add_argument("--b", default=os.path.join(ROOT, "data/out/submission_v22.zip"),
                    help="v22 变体（pick 值为 v22 时取它）")
    ap.add_argument("--pick", default=os.path.join(ROOT, "reports/variant_pick.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/out/submission_v23.zip"))
    a = ap.parse_args()

    pick = json.load(open(a.pick))
    za, zb = zipfile.ZipFile(a.a), zipfile.ZipFile(a.b)
    Pa = json.loads(za.read("predictions.json"))
    Pb = json.loads(zb.read("predictions.json"))
    assert set(Pa) == set(Pb), "两个变体的帧 id 不一致"

    out = {}
    for fid in Pb:
        rows = [r for r in Pa[fid] if pick.get(r["class"]) == "fused"]
        rows += [r for r in Pb[fid] if pick.get(r["class"]) != "fused"]
        out[fid] = sorted(rows, key=lambda x: -x["score"])

    src = zb
    names = zb.namelist()
    syn = [(n, zb.read(n)) for n in sorted(x for x in names if x.startswith("synthetic/"))]
    decl = zb.read("declaration.txt").decode("utf-8") if "declaration.txt" in names else ""

    ntot = sum(len(v) for v in out.values())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as o:
        for n, d in syn:
            o.writestr(n, d)
        o.writestr("predictions.json", json.dumps(out))
        if decl:
            o.writestr("declaration.txt", decl)
    print(f"融合类 {[k for k, v in pick.items() if v == 'fused']}")
    print(f"-> {a.out}  {ntot} 框 ({ntot/max(len(out),1):.0f}/帧)  "
          f"{os.path.getsize(a.out)/1024/1024:.1f} MB")
    _ = src


if __name__ == "__main__":
    main()
