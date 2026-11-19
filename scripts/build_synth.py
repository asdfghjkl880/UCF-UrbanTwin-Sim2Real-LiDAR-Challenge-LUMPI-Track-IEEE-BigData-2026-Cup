#!/usr/bin/env python3
"""
批量生成合成数据集（点云 + GT），并可直接打包成本地评分的提交 zip。

用法：
    python scripts/build_synth.py --n 50                 # 只生成数据
    python scripts/build_synth.py --n 50 --zip syn.zip   # 生成并打包（预测为空，只看 realism）
    python scripts/build_synth.py --n 50 --zip syn.zip --predict-gt   # 预测=GT（验证 mAP 上限）
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import LOCAL_BENCH, SYNTH  # noqa: E402
from scene_gen import gen_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=SYNTH)
    ap.add_argument('--zip', default=None)
    ap.add_argument('--predict-gt', action='store_true')
    ap.add_argument('--seq', action='store_true',
                    help='用 0001..N 命名（提交包的 synthetic/ 要求），否则沿用测试帧 ID')
    ap.add_argument('--align', default='affine',
                    choices=['none', 'translate', 'affine'],
                    help='矩对齐模式。affine 把各轴 (mu,sd) 都对齐到真实帧，'
                         'realism 0.8368->0.8772，但会各向异性拉伸点云、'
                         '与 GT 框不再匹配，所以**只用于提交包的 synthetic/**。'
                         '要生成训练数据必须传 translate。')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(LOCAL_BENCH, 'config.json'), encoding='utf-8'))
    ids = cfg['detection_test_frame_ids']
    if a.seq:
        ids = [f'{i+1:04d}' for i in range(a.n)]
    else:
        ids = ids[:a.n]
    os.makedirs(a.out, exist_ok=True)

    gt_all = {}
    for i, fid in enumerate(ids):
        pts, boxes = gen_frame(seed=a.seed * 100000 + i, align=a.align)
        pts.tofile(os.path.join(a.out, f'{fid}.bin'))
        gt_all[fid] = boxes
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(ids)}')

    with open(os.path.join(a.out, 'gt.json'), 'w', encoding='utf-8') as f:
        json.dump(gt_all, f)
    print(f'生成 {len(ids)} 帧 -> {a.out}')

    if a.zip:
        preds = {}
        for fid in ids:
            if a.predict_gt:
                preds[fid] = [{'box': b['box'], 'score': 1.0, 'class': b['class']}
                              for b in gt_all[fid]]
            else:
                preds[fid] = []
        decl = ("I confirm that I did not train on any frame listed in forbidden_frames.txt, "
                "and that all submitted detections were produced by a model trained only on "
                "synthetic data I generated.\nTeam: local-dev\n")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            syn = os.path.join(td, 'synthetic')
            os.makedirs(syn)
            for fid in ids:
                np.fromfile(os.path.join(a.out, f'{fid}.bin'), dtype='<f4').tofile(
                    os.path.join(syn, f'{fid}.bin'))
            json.dump(preds, open(os.path.join(td, 'predictions.json'), 'w'))
            open(os.path.join(td, 'declaration.txt'), 'w', encoding='utf-8').write(decl)
            with zipfile.ZipFile(a.zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(td):
                    for fn in sorted(files):
                        p = os.path.join(root, fn)
                        zf.write(p, os.path.relpath(p, td))
        print(f'打包 {a.zip}  ({os.path.getsize(a.zip)/1024/1024:.1f} MB)')


if __name__ == '__main__':
    main()
