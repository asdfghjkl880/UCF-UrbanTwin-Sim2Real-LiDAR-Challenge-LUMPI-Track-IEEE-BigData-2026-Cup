#!/usr/bin/env python3
"""
生成一个"本地开发用"的提交包，用来验证整条评分链路是否打通。

两种模式：
  --mode perfect   合成帧直接用参考帧本身、预测直接用 GT → 理论上
                   realism 和 mAP 都应接近满分。用来做冒烟测试。
  --mode dummy     随机点云 + 空预测 → 应该拿到接近 0 的分数。

用法：
    python scripts/make_dev_submission.py --mode perfect
    python scripts/make_dev_submission.py --mode dummy
输出：dev_submission.zip（工作区根目录）
"""
import argparse
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import LOCAL_BENCH  # noqa: E402
REF = LOCAL_BENCH
OUTZIP = os.path.join(ROOT, "dev_submission.zip")

DECL = (
    "I confirm that I did not train on any frame listed in forbidden_frames.txt, "
    "and that all submitted detections were produced by a model trained only on "
    "synthetic data I generated.\n"
    "Team: local-dev\nDate: 2026-11-08\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["perfect", "dummy"], default="perfect")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(REF, "config.json"), encoding="utf-8"))
    gt = json.load(open(os.path.join(REF, "detection_gt.json"), encoding="utf-8"))
    ids = cfg["detection_test_frame_ids"]

    with tempfile.TemporaryDirectory() as td:
        td = os.path.abspath(td)
        syn = os.path.join(td, "synthetic")
        os.makedirs(syn)

        if a.mode == "perfect":
            for fid in ids:
                src = os.path.join(REF, "realism_reference", f"{fid}.bin")
                shutil.copy2(src, os.path.join(syn, f"{fid}.bin"))
            preds = {fid: [{"box": b["box"], "score": 1.0, "class": b["class"]}
                           for b in gt.get(fid, [])] for fid in ids}
        else:
            rng = random.Random(0)
            for fid in ids:
                pts = [(rng.gauss(0, 12), rng.gauss(0, 12), rng.gauss(-1, 2))
                       for _ in range(2000)]
                with open(os.path.join(syn, f"{fid}.bin"), "wb") as f:
                    f.write(struct.pack("<%df" % (3 * len(pts)),
                                        *[v for p in pts for v in p]))
            preds = {fid: [] for fid in ids}

        json.dump(preds, open(os.path.join(td, "predictions.json"), "w"))
        open(os.path.join(td, "declaration.txt"), "w", encoding="utf-8").write(DECL)

        with zipfile.ZipFile(OUTZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(td):
                for fn in sorted(files):
                    p = os.path.join(root, fn)
                    zf.write(p, os.path.relpath(p, td))

    print(f"写出 {OUTZIP}  ({os.path.getsize(OUTZIP)/1024:.0f} KB)  模式={a.mode}")


if __name__ == "__main__":
    main()
