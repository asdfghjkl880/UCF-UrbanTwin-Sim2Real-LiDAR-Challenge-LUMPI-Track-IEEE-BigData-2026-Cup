#!/usr/bin/env python3
"""逐变换分别导出预测（不做任何合并），供 TTA 共识融合使用。

detect_tta 现在把多视角结果直接 NMS 掉，只保留最高分 —— 两视角"同时看到"
这件事本身就丢掉了。而"多视角一致"是一个**不需要标签**的置信度信号，
正好对当前排序效率只有 24%~46% 的弱类有用。

变换集合（都严格等价于刚体变换，框可无损还原）：
  p0  原始
  p1  y 轴镜像      (x, -y, -yaw)
  p2  x 轴镜像      (-x, y, -yaw)
  p3  180° 旋转     (-x, -y, yaw)      = p1 + p2
用法:
    py scripts/pred_passes.py --dir data/eval_m4_holdout --out data/out/_p4h.npz
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pp_config as C  # noqa: E402
from pred_ada import adabn, load_pts  # noqa: E402

# (名字, sx, sy, yaw_sign)
TRANSFORMS = [
    ("p0", 1.0, 1.0),
    ("p1", 1.0, -1.0),
    ("p2", -1.0, 1.0),
    ("p3", -1.0, -1.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/train/ckpt/pp_sbg3_best.pt")
    ap.add_argument("--dir", default="data/eval_m4_holdout")
    ap.add_argument("--ada", default="")
    ap.add_argument("--ada-n", type=int, default=24)
    ap.add_argument("--out", default="data/out/_passes.npz")
    ap.add_argument("--topk", type=int, default=3000)
    ap.add_argument("--thr", type=float, default=0.02)
    ap.add_argument("--iou-thr", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transforms", default="p0,p1,p2,p3")
    ap.add_argument("--test", action="store_true",
                    help="改用官方 50 个测试帧（无 GT，仅 .bin）")
    a = ap.parse_args()

    import torch
    import pp_model as M
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.PointPillars().to(dev).eval()
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    print(f"载入 {a.ckpt}  device={dev}", flush=True)

    if a.ada:
        files = sorted(glob.glob(os.path.join(a.ada, "*.bin")))
        rng = np.random.default_rng(0)
        if len(files) > a.ada_n:
            files = [files[i] for i in
                     sorted(rng.choice(len(files), a.ada_n, replace=False))]
        nb, used = adabn(model, dev, files)
        print(f"AdaBN: {nb} 层 / {used} 帧", flush=True)

    use = [t for t in TRANSFORMS if t[0] in a.transforms.split(",")]
    if a.test:
        from predict_pp import load_frame_ids, read_bin
        frames = [(fid, read_bin(os.path.join(C.TEST_FRAMES_DIR, f"{fid}.bin")),
                   np.zeros((0, 8)))
                  for fid in load_frame_ids()]
    else:
        from diag_rank import _frames_from_dir
        frames = list(_frames_from_dir(a.dir, a.limit))
    out = {}
    for i, (fid, pts, gt) in enumerate(frames):
        rec = {}
        for nm, sx, sy in use:
            q = np.ascontiguousarray(pts.copy(), dtype=np.float32)
            q[:, 0] *= sx
            q[:, 1] *= sy
            b, s, c = M.detect(model, q, dev, a.thr, a.iou_thr, topk=a.topk)
            if len(b):
                b = b.copy()
                b[:, 0] *= sx
                b[:, 1] *= sy
                # 镜像一个轴时 yaw -> -yaw；镜像两个轴（180°）时 yaw 不变
                if sx * sy < 0:
                    b[:, 6] = -b[:, 6]
            rec[nm] = (b.astype(np.float32), s.astype(np.float32),
                       c.astype(np.int64))
        out[fid] = (rec, gt)
        if i % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] {fid} "
                  + " ".join(f"{k}:{len(v[0])}" for k, v in rec.items()), flush=True)
    np.save(a.out, out, allow_pickle=True)
    print("写出", a.out)


if __name__ == "__main__":
    main()
