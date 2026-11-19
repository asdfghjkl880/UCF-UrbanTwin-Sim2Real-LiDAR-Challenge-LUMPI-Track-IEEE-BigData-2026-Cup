"""只算 realism 的快速评估器（跳过 200 帧检测推理）。

直接调用主办方 metrics.compute_realism，口径与 score.py 完全一致：
拼 50 帧 -> 按 ±40m/±5m 掩码 -> 子采样 10000 点 -> CD / MMD / EMD / FPD。

用法：
    python scripts/eval_realism.py --synth data/synthetic_probe2 \
        --ref data/eval_m4_test/realism_reference
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'lumpi_starting_kit', 'scoring_program'))
import metrics  # noqa: E402

PC_RANGE = (-40.0, -40.0, -5.0, 40.0, 40.0, 5.0)
# score.py 里的归一化区间
NORMS = {'CD': (2.0, 30.0), 'MMD': (0.0, 0.5), 'EMD': (0.0, 5.0), 'FPD': (0.0, 50.0)}


def load(d, limit=50):
    out = []
    for p in sorted(glob.glob(os.path.join(d, '*.bin')))[:limit]:
        out.append(np.fromfile(p, dtype=np.float32).reshape(-1, 3))
    return out


def normalize(v, lo, hi, lower_is_better):
    x = (v - lo) / (hi - lo)
    x = min(max(x, 0.0), 1.0)
    return 1.0 - x if lower_is_better else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', required=True)
    ap.add_argument('--ref', default=os.path.join(ROOT, 'data', 'eval_m4_test',
                                                  'realism_reference'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    syn = load(a.synth)
    real = load(a.ref)
    if not syn or not real:
        raise SystemExit('帧为空')
    if len(syn) != len(real):
        print(f'  [警告] 合成 {len(syn)} 帧 vs 参考 {len(real)} 帧，数量不等')

    r = metrics.compute_realism(syn, real, PC_RANGE, seed=1234)
    norm = float(np.mean([
        normalize(r['CD'], *NORMS['CD'], True),
        normalize(r['MMD'], *NORMS['MMD'], True),
        normalize(r['EMD'], *NORMS['EMD'], True),
        normalize(r['FPD'], *NORMS['FPD'], True),
    ]))
    print(f'{"tag":<24}{"CD":>7}{"MMD":>10}{"EMD":>8}{"FPD":>8}{"realism":>9}')
    print(f'{a.tag or os.path.basename(a.synth):<24}'
          f'{r["CD"]:>7.3f}{r["MMD"]:>10.5f}{r["EMD"]:>8.3f}{r["FPD"]:>8.3f}'
          f'{norm:>9.4f}')
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({'realism': r, 'realism_norm': norm, 'tag': a.tag}, f, indent=2)


if __name__ == '__main__':
    main()
