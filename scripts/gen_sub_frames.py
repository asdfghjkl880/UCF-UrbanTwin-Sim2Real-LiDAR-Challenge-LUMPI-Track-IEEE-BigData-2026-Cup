"""导出提交用的合成帧（不参与检测评分，只用于 realism）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import scene_gen as G  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/synthetic_v9')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=555000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f'BG_MODE={G.BG_MODE} OBJ_SRC={G.OBJ_SRC} SHADOW={G.SHADOW_BOX} '
          f'DENS={G.OBJ_DENSITY} cal={G._cal()}')
    for i in range(a.n):
        pts, _ = G.gen_frame(seed=a.seed + i)
        pts = np.asarray(pts, np.float32)
        pts.tofile(os.path.join(a.out, f'synthetic_{i:04d}.bin'))
    print(f'-> {a.out} {a.n} 帧')


if __name__ == '__main__':
    main()
