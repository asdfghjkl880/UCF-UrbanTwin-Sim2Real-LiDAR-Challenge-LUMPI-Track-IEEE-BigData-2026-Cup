#!/usr/bin/env python3
"""按官方评分器的 validate_submission() 逐条校验提交包，不需要 GT。

直接把包解压到临时目录，调用主办方 scoring_program/score.py 里的
validate_submission()，再补几项 score.py 不查但 submission_format.md 要求的检查
（解压总大小、单帧点数、score 取值范围、box 数值有限性）。

用法：
    python scripts/check_submission.py data/out/submission.zip
"""
import json
import os
import sys
import tempfile
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, 'lumpi_starting_kit')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'data', 'out', 'submission.zip')
    cfg = json.load(open(os.path.join(KIT, 'config.json'), encoding='utf-8')) \
        if os.path.exists(os.path.join(KIT, 'config.json')) else None
    if cfg is None:
        # 起跑包没带 config.json 时，用本地基准的（class_names / 范围都在里面）
        cfg = json.load(open(os.path.join(ROOT, 'data', 'eval_m4_test',
                                         'config.json'), encoding='utf-8'))
    cfg['detection_test_frame_ids'] = json.load(
        open(os.path.join(KIT, 'detection_test_frame_ids.json'), encoding='utf-8'))
    cfg.setdefault('n_realism_frames', 50)

    ok, bad = [], []

    def chk(cond, msg):
        (ok if cond else bad).append(msg)

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(path) as z:
            z.extractall(td)
        # score.py 通过模块级 RES_DIR 定位解压目录，这里临时改掉
        sys.path.insert(0, os.path.join(KIT, 'scoring_program'))
        import score as S
        S.RES_DIR = __import__('pathlib').Path(td)

        try:
            syn_files, preds = S.validate_submission(cfg)
            chk(True, 'validate_submission() 通过（声明 / 50 帧 / 帧 id / box 长度）')
        except SystemExit:
            chk(False, 'validate_submission() 拒绝，详见上方 FATAL 信息')
            syn_files, preds = [], {}

        # ---- submission_format.md 里 score.py 没查的部分 ----
        tot = sum(os.path.getsize(os.path.join(r, f))
                  for r, _, fs in os.walk(td) for f in fs)
        chk(tot < 75e6, f'解压总大小 {tot/1e6:.1f} MB < 75 MB')

        npts = []
        for f in syn_files:
            a = np.fromfile(f, dtype=np.float32)
            n = a.size // 3 if a.size % 3 == 0 else a.size // 4
            npts.append(n)
            chk(f.stat().st_size <= 1_500_000,
                f'{os.path.basename(f)} {f.stat().st_size/1024:.0f} KB ≤ 1.5 MB')
        if npts:
            chk(max(npts) <= 125000,
                f'单帧最大点数 {max(npts)} ≤ 125,000（最小 {min(npts)}）')

        allb, alls, allc = [], [], set()
        for fid, dets in preds.items():
            for d in dets:
                allb.append(d['box'])
                alls.append(d['score'])
                allc.add(d.get('class', 'Car'))
        B = np.array(allb, np.float64) if allb else np.zeros((0, 7))
        Sc = np.array(alls, np.float64) if alls else np.zeros(0)
        valid = set(cfg.get('class_names', []))
        chk(allc <= valid,
            f'出现过的类别都在 config.class_names 内（{len(allc)} 类：'
            f'{", ".join(sorted(allc))}）')
        chk(np.isfinite(B).all(), 'box 全部为有限值')
        chk(np.isfinite(Sc).all(), 'score 全部为有限值')
        chk(((Sc >= 0) & (Sc <= 1)).all() if len(Sc) else True,
            f'score 全在 [0,1]（min {Sc.min():.3f} max {Sc.max():.3f}）')
        chk((B[:, 3:6] > 0).all() if len(B) else True, 'box 尺寸 dx/dy/dz 全为正')
        # 官方格式并不要求框中心落在 point_cloud_range 内：基准 GT 本身就有 17%
        # 的框中心在 ±40 m 之外（最远 82.9 m）。这里只作提示，不算失败。
        if len(B):
            out = int(((np.abs(B[:, 0]) > 40) | (np.abs(B[:, 1]) > 40)).sum())
            print(f'  [提示] {out}/{len(B)} 个框中心超出 ±40 m '
                  f'({100*out/len(B):.2f}%)，官方允许，保留可多覆盖远处 GT')

        # 声明内容：官方允许 declaration.txt，也接受扫描/排版后的 declaration.pdf
        decl_txt = os.path.join(td, 'declaration.txt')
        decl_pdf = os.path.join(td, 'declaration.pdf')
        if os.path.exists(decl_txt):
            decl = open(decl_txt, encoding='utf-8').read()
            need = ('I confirm that I did not train on any frame listed in',
                    'synthetic data I generated',
                    'Team / individual name:', 'Date (YYYY-MM-DD):',
                    'Submission tag:')
            for s in need:
                chk(s.lower() in decl.lower(), f'声明包含 “{s}”')
        elif os.path.exists(decl_pdf):
            data = open(decl_pdf, 'rb').read()
            chk(data[:5] == b'%PDF-', 'declaration.pdf 是合法 PDF（头部 %PDF-）')
            chk(len(data) > 20000, f'declaration.pdf 非空白（{len(data)/1024:.1f} KB）')
            print('  [提示] declaration.pdf 为排版版，关键字逐条校验跳过'
                  '（生成时已按模板校验过必需语句）')
        else:
            chk(False, '存在 declaration.txt 或 declaration.pdf')

    print()
    for m in ok:
        if not m.startswith('类别'):
            print(f'  [OK]   {m}')
    for m in bad:
        print(f'  [FAIL] {m}')
    # 类别检查是逐条重复的，单独汇总
    print()
    print(f'通过 {len(ok)} 项，失败 {len(bad)} 项')
    print('结论：' + ('可以直接提交' if not bad else '还有问题，见上'))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
