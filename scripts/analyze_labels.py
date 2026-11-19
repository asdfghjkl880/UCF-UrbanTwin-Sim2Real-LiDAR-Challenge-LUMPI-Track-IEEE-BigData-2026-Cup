#!/usr/bin/env python3
"""
统计 LUMPI 真实标注：8 个类别的目标尺寸 / 数量 / 朝向分布。
这是合成数据的"图纸"——你的合成场景要照着这些数字造。

用法：
    python scripts/analyze_labels.py                 # 全部 measurement
    python scripts/analyze_labels.py Measurement5    # 只看某个（推荐，测试帧来自 M5）
输出：控制台报告 + reports/label_stats.md
"""
import csv
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import LABEL_DIR  # noqa: E402
OUT = os.path.join(ROOT, "reports", "label_stats.md")

CLASS = {0: "Person", 1: "Car", 2: "Bicycle", 3: "Motorcycle",
         4: "Bus", 5: "Truck", 6: "Unknown"}
ORDER = ["Person", "Car", "Bicycle", "Motorcycle", "Bus", "Truck", "Van", "Unknown"]

COLS = ["time", "obj", "r2x", "r2y", "r2w", "r2h", "score", "cls", "vis",
        "cx", "cy", "cz", "L", "W", "H", "heading"]


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main():
    targets = sys.argv[1:] or ["Measurment0", "Measurement1", "Measurement2",
                               "Measurement3", "Measurement4", "Measurement5",
                               "Measurement6"]
    lines = []
    for m in targets:
        path = os.path.join(LABEL_DIR, m, "Label.csv")
        if not os.path.exists(path):
            print(f"!! 找不到 {path}")
            continue

        dims = defaultdict(lambda: {"L": [], "W": [], "H": [], "z": [], "hd": [], "vis": []})
        obj_ids = defaultdict(set)
        rows_per_cls = Counter()
        frames = set()
        frames_by_cls = defaultdict(set)
        score_cnt = Counter()

        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            rd = csv.reader(f)
            next(rd)
            for r in rd:
                if len(r) < 16:
                    continue
                try:
                    t = float(r[0]); oid = r[1].strip(); score = float(r[6])
                    cls = int(float(r[7])); vis = float(r[8])
                    cx, cy, cz = float(r[9]), float(r[10]), float(r[11])
                    L, W, H = float(r[12]), float(r[13]), float(r[14])
                    hd = float(r[15])
                except ValueError:
                    continue
                frames.add(t)
                score_cnt[score] += 1
                name = CLASS.get(cls)
                if name is None:
                    continue
                d = dims[name]
                d["L"].append(L); d["W"].append(W); d["H"].append(H)
                d["z"].append(cz); d["hd"].append(hd); d["vis"].append(vis)
                obj_ids[name].add(oid)
                rows_per_cls[name] += 1
                frames_by_cls[name].add(t)

        nf = len(frames)
        head = [f"\n## {m}", "",
                f"- 总帧数（按标注时间戳计）: **{nf}**",
                f"- 标注行数: **{sum(rows_per_cls.values())}**",
                f"- score=4（人工校正）占比: **{score_cnt[4.0] / max(1, sum(score_cnt.values())) * 100:.1f}%**",
                "",
                "| 类别 | 实例数 | 唯一目标数 | 出现帧数 | 每帧平均 | 长 L (m) | 宽 W (m) | 高 H (m) | z 中心 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name in ORDER:
            if name not in dims:
                head.append(f"| {name} | 0 | 0 | 0 | – | – | – | – | – |")
                continue
            d = dims[name]
            n = rows_per_cls[name]
            head.append(
                f"| {name} | {n} | {len(obj_ids[name])} | {len(frames_by_cls[name])} "
                f"| {n / max(1, len(frames_by_cls[name])):.2f} "
                f"| {statistics.median(d['L']):.2f} | {statistics.median(d['W']):.2f} "
                f"| {statistics.median(d['H']):.2f} | {statistics.median(d['z']):.2f} |")

        head += ["", "尺寸分位数（L × W × H，单位 m）：", "",
                 "| 类别 | p5 | p50 | p95 | 标准差 L |", "|---|---|---|---|---|"]
        for name in ORDER:
            if name not in dims:
                continue
            L = dims[name]["L"]
            head.append(f"| {name} | {pct(L, .05):.2f}×{pct(dims[name]['W'], .05):.2f}×{pct(dims[name]['H'], .05):.2f} "
                        f"| {pct(L, .5):.2f}×{pct(dims[name]['W'], .5):.2f}×{pct(dims[name]['H'], .5):.2f} "
                        f"| {pct(L, .95):.2f}×{pct(dims[name]['W'], .95):.2f}×{pct(dims[name]['H'], .95):.2f} "
                        f"| {statistics.pstdev(L):.2f} |")

        lines += head
        print("\n".join(head))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# LUMPI 真实标注统计\n")
        f.write("\n由 `scripts/analyze_labels.py` 生成。合成场景应照此分布构造目标。\n")
        f.write("\n".join(lines))
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
