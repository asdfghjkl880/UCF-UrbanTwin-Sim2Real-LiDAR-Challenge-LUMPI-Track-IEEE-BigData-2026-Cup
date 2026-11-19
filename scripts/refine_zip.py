#!/usr/bin/env python3
"""在提交包上做几何精修（geo_fit 多候选 + 点支撑择优）。

用法:
    py scripts/refine_zip.py --in data/out/_test_raw.zip \
        --clouds lumpi_starting_kit/detection_test_frames \
        --out data/out/_test_ref.zip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo_fit as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clouds', default='lumpi_starting_kit/detection_test_frames')
    a = ap.parse_args()

    z = zipfile.ZipFile(a.inp)
    P = json.loads(z.read('predictions.json'))
    decl = z.read('declaration.txt').decode('utf-8')
    syn = sorted(x for x in z.namelist() if x.startswith('synthetic/'))

    n_in = n_out = 0
    for fi, fid in enumerate(sorted(P)):
        dets = P[fid]
        if not dets:
            continue
        p = os.path.join(a.clouds, fid + '.bin')
        if not os.path.exists(p):
            continue
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 3).astype(np.float64)
        if len(pts) == 0:
            continue
        tree = cKDTree(pts)
        B = np.array([d['box'] for d in dets], np.float64)
        cls = np.array([G.NAMES.index(d.get('class', 'Car')) if d.get('class')
                        in G.NAMES else 1 for d in dets], np.int64)
        R = G.fit_frame(pts, B, cls, tree=tree)
        n_in += len(B)
        for i, d in enumerate(dets):
            d['box'] = [float(v) for v in R[i]]
        n_out += len(R)
        if fi % 10 == 0:
            print(f'  [{fi+1}] {fid} {len(dets)} 框', flush=True)

    with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_DEFLATED) as o:
        for nm in syn:
            o.writestr(nm, z.read(nm))
        o.writestr('predictions.json', json.dumps(P))
        o.writestr('declaration.txt', decl)
    print(f'精修完成 {n_in} -> {n_out} 框，写出 {a.out}')


if __name__ == '__main__':
    main()
