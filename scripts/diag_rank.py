#!/usr/bin/env python3
"""诊断 mAP 的两个乘数：召回上限（框几何）与排序效率（分数）。

官方 AP 是 11 点插值（recall 取 0,0.1,...,1.0），所以
    AP <= (可召回到的最高 recall 档位)/11 * 峰值精度
召回达不到 1.0 时，AP 有硬天花板。本脚本把 mAP 拆成：
  * oracle recall@0.5   —— 完美排序、任意多框时能召回的 GT 比例（框几何上限）
  * AP / oracle_recall  —— 排序效率
再按距离分桶、按"换中心/换朝向/换尺寸"分解，定位召回损失来源。

输出 cache（--cache）便于反复分析，不重复推理。
用法:
    py scripts/diag_rank.py --ckpt data/train/ckpt/pp_w1_best.pt --cache data/out/_rank_w1.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'lumpi_starting_kit', 'scoring_program'))
import pp_config as C  # noqa: E402
import pp_model as M  # noqa: E402
import detection_eval as DE  # noqa: E402


def iou3d(a, b):
    return float(DE._iou_3d(np.asarray(a, float), np.asarray(b, float)))


def _frames_from_dir(dirpath, limit=0):
    """统一取帧：既支持 .bin/.json 帧目录，也支持 shard_*.npy + _meta.json。"""
    js = [f for f in os.listdir(dirpath)
          if f.endswith('.json') and f[:-5].isdigit()]
    if js:
        fids = sorted(f[:-5] for f in js)
        if limit:
            idx = np.linspace(0, len(fids) - 1, min(limit, len(fids))).astype(int)
            fids = [fids[i] for i in sorted(set(idx.tolist()))]
        for fid in fids:
            pts = np.fromfile(os.path.join(dirpath, fid + '.bin'),
                              dtype=np.float32).reshape(-1, 3)
            gt = [r for r in json.load(open(os.path.join(dirpath, fid + '.json')))
                  if len(r) == 8]
            yield fid, pts, np.array(gt, np.float32).reshape(-1, 8)
        return
    meta_paths = sorted(glob.glob(os.path.join(dirpath, '*_meta.json')))
    if not meta_paths:
        raise FileNotFoundError(f'{dirpath} 里没有 .json 帧也没有 *_meta.json 分片')
    k = 0
    for mp in meta_paths:
        meta = json.load(open(mp))
        arr = np.load(mp.replace('_meta.json', '.npy'), mmap_mode='r')
        off = np.array(meta['offsets'])
        for i, labs in enumerate(meta['labels']):
            if limit and k >= limit:
                return
            pts = np.asarray(arr[off[i]:off[i + 1]], np.float32)
            gt = np.array([[l[0]] + list(l[1:8]) for l in labs],
                          np.float32).reshape(-1, 8)
            yield f's{k:05d}', pts, gt
            k += 1


def predict_dir(ckpt, dirpath, dev, score_thr, iou_thr, topk, limit=0,
                tta=False, dup=None):
    model = M.PointPillars().to(dev)
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(st['model'])
    model.eval()
    out = {}
    n = 0
    for k, (fid, pts, gt) in enumerate(_frames_from_dir(dirpath, limit)):
        if tta or dup:
            b, s, c = M.detect_tta(model, pts, dev, score_thr, iou_thr,
                                   topk=topk, flips=((1,) if tta else ()),
                                   dup=dup)
        else:
            b, s, c = M.detect(model, pts, dev, score_thr, iou_thr, topk=topk)
        out[fid] = (b.astype(np.float32), s.astype(np.float32),
                    c.astype(np.int64), gt)
        n = k + 1
        if n % 20 == 0:
            print(f'  {n}', flush=True)
    return out


def load_cache(path):
    z = np.load(path, allow_pickle=True)
    return {k: tuple(z[k]) for k in z.files}


def save_cache(path, preds):
    np.savez_compressed(path, **{k: np.array(v, dtype=object) for k, v in preds.items()})


def ap11(preds, gt):
    res = DE.evaluate_3d_detection(preds, gt, C.EVAL_CLASS_NAMES,
                                   iou_thresholds=(0.5,), iou_type='3d')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='data/train/ckpt/pp_w1_best.pt')
    ap.add_argument('--val', default='data/eval_m4_holdout')
    ap.add_argument('--cache', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--thr', type=float, default=0.10)
    ap.add_argument('--iou', type=float, default=0.25)
    ap.add_argument('--topk', type=int, default=200)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--tta', action='store_true', help='镜像 TTA')
    ap.add_argument('--dup', default='', help='如 Van:Car:0.9,Motorcycle:Bicycle:0.8')
    a = ap.parse_args()

    dev = torch.device('cpu' if a.cpu else 'cuda')
    if a.cache and os.path.exists(a.cache):
        print(f'读取 cache {a.cache}')
        preds = load_cache(a.cache)
    else:
        dup = {}
        for it in (a.dup.split(',') if a.dup else []):
            if not it:
                continue
            t, src, fac = it.split(':')
            dup[t] = (src, float(fac))
        preds = predict_dir(a.ckpt, a.val, dev, a.thr, a.iou, a.topk, a.limit,
                            tta=a.tta, dup=dup or None)
        if a.cache:
            save_cache(a.cache, preds)
            print(f'写出 {a.cache}')

    # ---- 官方口径 AP ----
    ev_pred, ev_gt = {}, {}
    for fid, (b, s, c, g) in preds.items():
        ev_pred[fid] = [{'box': [float(v) for v in bb],
                         'score': float(ss), 'class': C.CLASS_NAMES[int(cc)]}
                        for bb, ss, cc in zip(b, s, c)]
        ev_gt[fid] = [{'box': [float(v) for v in r[1:8]],
                       'class': C.CLASS_NAMES[int(r[0])]} for r in g]
    res = ap11(ev_pred, ev_gt)
    names = C.CLASS_NAMES

    print()
    print(f'{"class":<11}{"n_gt":>6}{"n_pred":>8}{"AP":>7}{"oracleR":>9}'
          f'{"AP/oR":>7}{"medIoU":>8}{"d_center":>9}{"d_hd":>7}')
    tot_ap, tot_or = 0.0, 0.0
    for ci, nm in enumerate(names):
        n_gt = n_pred = n_or = 0
        best_ious, dc, dh = [], [], []
        for fid, (b, s, c, g) in preds.items():
            gi = g[:, 0] == ci
            pi = c == ci
            n_gt += int(gi.sum())
            n_pred += int(pi.sum())
            if not gi.any():
                continue
            gg = g[gi]
            bb = b[pi]
            for gb in gg:
                if len(bb) == 0:
                    best_ious.append(0.0)
                    continue
                ious = [iou3d(p, gb[1:8]) for p in bb]
                j = int(np.argmax(ious))
                best_ious.append(ious[j])
                if ious[j] >= 0.5:
                    n_or += 1
                    dc.append(float(np.hypot(bb[j][0] - gb[1], bb[j][1] - gb[2])))
                    dh.append(float(abs(((bb[j][6] - gb[7]) + np.pi / 2) % np.pi
                                        - np.pi / 2)))
        oR = n_or / max(n_gt, 1)
        apv = res['per_class'][nm]['AP_0.5_R40']
        tot_ap += apv
        tot_or += oR
        bi = np.array(best_ious) if best_ious else np.zeros(1)
        print(f'{nm:<11}{n_gt:>6}{n_pred:>8}{apv:>7.3f}{oR:>9.3f}'
              f'{(apv/max(oR,1e-6)):>7.2f}{np.median(bi):>8.2f}'
              f'{(np.median(dc) if dc else -1):>9.2f}{(np.median(dh) if dh else -1):>7.2f}')
    print(f'{"mAP":<11}{"":>6}{"":>8}{tot_ap/7:>7.4f}{tot_or/7:>9.3f}')

    # ---- 距离分桶召回 ----
    print('\n按距离分桶的 oracle recall@0.5（桶内 GT 能被任一同类预测命中）')
    edges = [0, 10, 20, 30, 40, 60]
    print(f'{"class":<11}' + ''.join(f'{f"{edges[i]}-{edges[i+1]}":>10}'
                                     for i in range(len(edges) - 1)))
    for ci, nm in enumerate(names):
        row = []
        for i in range(len(edges) - 1):
            ng = nhit = 0
            for fid, (b, s, c, g) in preds.items():
                gi = (g[:, 0] == ci)
                r = np.hypot(g[:, 1], g[:, 2])
                gi &= (r >= edges[i]) & (r < edges[i + 1])
                if not gi.any():
                    continue
                pi = c == ci
                bb = b[pi]
                for gb in g[gi]:
                    ng += 1
                    if len(bb) == 0:
                        continue
                    if max(iou3d(p, gb[1:8]) for p in bb) >= 0.5:
                        nhit += 1
            row.append(f'{nhit}/{ng}' if ng else '-')
        print(f'{nm:<11}' + ''.join(f'{v:>10}' for v in row))

    # ---- 三项替换分解 ----
    print('\n单项替换后的 oracle recall@0.5（centre / heading / size 分别换成 GT）')
    print(f'{"class":<11}{"orig":>7}{"+ctr":>7}{"+hd":>7}{"+sz":>7}'
          f'{"ctr+hd":>8}{"all":>7}')
    for ci, nm in enumerate(names):
        cnt = {'o': 0, 'c': 0, 'h': 0, 's': 0, 'ch': 0, 'a': 0}
        ng = 0
        for fid, (b, s, c_, g) in preds.items():
            gi = g[:, 0] == ci
            if not gi.any():
                continue
            bb = b[c_ == ci]
            for gb in g[gi]:
                ng += 1
                if len(bb) == 0:
                    continue
                gbox = gb[1:8]
                ious = [iou3d(p, gbox) for p in bb]
                j = int(np.argmax(ious))
                p = bb[j].copy()
                ctr = np.array([gbox[0], gbox[1], gbox[2], p[3], p[4], p[5], p[6]])
                hd = np.array([p[0], p[1], p[2], p[3], p[4], p[5], gbox[6]])
                sz = np.array([p[0], p[1], p[2], gbox[3], gbox[4], gbox[5], p[6]])
                ch = np.array([gbox[0], gbox[1], gbox[2], p[3], p[4], p[5], gbox[6]])
                for k, v in (('o', p), ('c', ctr), ('h', hd), ('s', sz),
                             ('ch', ch), ('a', gbox)):
                    cnt[k] += iou3d(v, gbox) >= 0.5
        if not ng:
            continue
        print(f'{nm:<11}' + ''.join(f'{cnt[k]/ng:>7.3f}' for k in 'ochs')
              + f'{cnt["ch"]/ng:>8.3f}{cnt["a"]/ng:>7.3f}')

    # ---- 分数分辨能力 ----
    print('\n分数分辨力（同类内 TP 与 FP 的分数分布；AUC 越高越好）')
    for ci, nm in enumerate(names):
        tp, fp = [], []
        for fid, (b, s, c_, g) in preds.items():
            bb = b[c_ == ci]
            ss = s[c_ == ci]
            gi = g[:, 0] == ci
            gg = g[gi]
            used = set()
            for j, p in enumerate(bb):
                hit = -1
                bi = -1
                for k, gb in enumerate(gg):
                    if k in used:
                        continue
                    v = iou3d(p, gb[1:8])
                    if v > bi:
                        bi, hit = v, k
                if bi >= 0.5:
                    tp.append(ss[j])
                    used.add(hit)
                else:
                    fp.append(ss[j])
        if not tp and not fp:
            continue
        tp, fp = np.array(tp), np.array(fp)
        auc = np.mean([1.0 if t > f else 0.5 if t == f else 0.0
                       for t in tp[:400] for f in fp[:400]]) if len(tp) and len(fp) else 0
        print(f'{nm:<11} TP {len(tp):>5} p50={np.median(tp) if len(tp) else 0:.3f}  '
              f'FP {len(fp):>5} p50={np.median(fp) if len(fp) else 0:.3f}  '
              f'FP>=minTP {int((fp >= (tp.min() if len(tp) else 1)).sum()):>5}  AUC={auc:.3f}')


if __name__ == '__main__':
    main()
