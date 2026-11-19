#!/usr/bin/env python3
"""
构建"合法"的本地基准集（不碰 forbidden_frames.txt 里的 100 帧）。

数据来源：test_data/test_data/Measurement4/ —— 真实点云帧 000000~000030
（时间 0.0~3.0s），这些帧号远小于禁用区间 6497~8109，完全干净。

产出 reference_data_local/：
  config.json                  n_realism_frames=31，帧 ID 就是 000000..000030
  realism_reference/*.bin      31 帧真实点云（裁 ±40m，子采样 5 万点）
  detection_gt.json            31 帧的真实标注框（来自 Measurement4 的 Label.csv）

之后 local_eval.py 就能给出**真实的** realism 分数和**真实的** mAP。
"""
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lumpi_io import crop_box, read_ply_xyz, subsample, write_bin_xyz  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_DATA, LABEL_DIR, LOCAL_BENCH, KIT  # noqa: E402

SRC = os.path.join(TEST_DATA, "Measurement4", "lidar")
LABEL = os.path.join(LABEL_DIR, "Measurement4", "Label.csv")
OUT = LOCAL_BENCH

FPS = 10.0
N_POINTS = 50000
CLASS = {0: "Person", 1: "Car", 2: "Bicycle", 3: "Motorcycle",
         4: "Bus", 5: "Truck", 6: "Van"}


def load_gt(path):
    gt = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            if len(r) < 16:
                continue
            try:
                t = float(r[0]); cls = int(float(r[7]))
                box = [float(r[9]), float(r[10]), float(r[11]),
                       float(r[12]), float(r[13]), float(r[14]), float(r[15])]
                score = float(r[6])
            except ValueError:
                continue
            name = CLASS.get(cls)
            if name is None:
                continue
            gt.setdefault(round(t, 2), []).append(
                {"box": [round(v, 3) for v in box], "class": name, "score": score})
    return gt


def main():
    frames = sorted(f for f in os.listdir(SRC) if f.endswith(".ply"))
    if not frames:
        raise SystemExit(f"没找到点云：{SRC}")
    print(f"源帧 {len(frames)} 个：{frames[0]} ~ {frames[-1]}")

    os.makedirs(os.path.join(OUT, "realism_reference"), exist_ok=True)
    ids, gt_out = [], {}
    gt_all = load_gt(LABEL)

    for fn in frames:
        fid = os.path.splitext(fn)[0]          # 000000
        idx = int(fid)
        t = round(idx / FPS, 2)
        pts = crop_box(read_ply_xyz(os.path.join(SRC, fn)))
        pts = subsample(pts, N_POINTS, seed=1234)
        write_bin_xyz(os.path.join(OUT, "realism_reference", f"{fid}.bin"), pts)
        ids.append(fid)
        boxes = gt_all.get(t, [])
        gt_out[fid] = [{"box": b["box"], "class": b["class"]} for b in boxes]
        print(f"  {fid}  t={t:5.2f}s  裁盒后→{len(pts)} 点  标注框 {len(boxes)} 个")

    with open(os.path.join(KIT, "reference_data_dummy", "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["n_realism_frames"] = len(ids)
    cfg["detection_test_frame_ids"] = ids
    (open(os.path.join(OUT, "config.json"), "w", encoding="utf-8")
     .write(json.dumps(cfg, indent=2, ensure_ascii=False)))
    (open(os.path.join(OUT, "detection_gt.json"), "w", encoding="utf-8")
     .write(json.dumps(gt_out, indent=1, ensure_ascii=False)))

    tot = sum(len(v) for v in gt_out.values())
    print(f"\n完成：{len(ids)} 帧，共 {tot} 个标注框 → {OUT}")
    from collections import Counter
    c = Counter(b["class"] for v in gt_out.values() for b in v)
    print("类别分布：", dict(c))


if __name__ == "__main__":
    main()
