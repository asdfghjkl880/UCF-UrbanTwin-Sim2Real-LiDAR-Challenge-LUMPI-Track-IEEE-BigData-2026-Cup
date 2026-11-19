#!/usr/bin/env python3
"""
PointPillars 训练脚本。

数据：合成集（数千帧）为主，真实集（155 帧，M1/M2/M3/M5/M6）混入做域适配。
验证：Measurement4 的 31 帧真实标注（与 50 个测试帧同传感器同路口，帧号不重叠），
      直接用官方 detection_eval 算 mAP@3DIoU0.5。

用法：
    python scripts/train_pp.py --epochs 40 --bs 4 --workers 6
    python scripts/train_pp.py --resume
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pp_config as C  # noqa: E402
import pp_data as D  # noqa: E402
import pp_model as M  # noqa: E402

KIT = os.path.join(os.path.dirname(C.DATA), 'lumpi_starting_kit')
sys.path.insert(0, os.path.join(KIT, 'scoring_program'))
from detection_eval import evaluate_3d_detection  # noqa: E402


def to_eval_format(boxes, scores, cls_ids, names=C.CLASS_NAMES):
    # 必须用模型侧的 7 类名，索引 6 = Van（2026-11-09 更正，之前判成 Unknown）。
    # 评分器认的是 config.json 的 8 个名字，其中 Van 也在索引 6，正好对上；
    # Unknown 因 n_gt=0 被自动跳过。名字必须逐字一致，写错一个该类 AP 直接归零。
    return [{'box': [float(v) for v in b], 'score': float(s), 'class': names[c]}
            for b, s, c in zip(boxes, scores, cls_ids)]


def make_loader(dirs, augment, bs, workers, shuffle):
    ds = D.FrameDataset(dirs, augment=augment)
    return ds, DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=workers,
        collate_fn=D.collate, pin_memory=True,
        persistent_workers=(workers > 0), drop_last=shuffle)


def load_gt(dirpath):
    """验证目录里的 .json 就是 GT。支持两种写法：
      * 行列表 [cls, cx,cy,cz,L,W,H,hd]（val_real / M4 导出）
      * 字典 {'box': [cx..hd], 'class': 'Person'}（eval_m4_test 导出）
    config.json / *_meta.json 不是标注，跳过。
    """
    out = {}
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith('.json') or fn.endswith('_meta.json') or fn == 'config.json':
            continue
        if fn == 'detection_gt.json':
            continue    # 聚合标注文件，不是单帧标注
        fid = fn[:-5]
        rows = json.load(open(os.path.join(dirpath, fn)))
        if rows and isinstance(rows[0], dict):
            out[fid] = [{'box': [float(v) for v in r['box']],
                         'class': str(r['class'])} for r in rows]
            continue
        out[fid] = [{'box': [float(v) for v in r[1:8]], 'class': C.CLASS_NAMES[int(r[0])]}
                    for r in rows]
    return out


def _shard_items(dirpath):
    """把 shard 格式的目录展开成 [(fid, stem, a, b, labels), ...]。"""
    shards = sorted(f for f in os.listdir(dirpath)
                    if f.startswith('shard_') and f.endswith('.npy'))
    if not shards:
        return None
    items = []
    for fn in shards:
        stem = fn[:-4]
        with open(os.path.join(dirpath, stem + '_meta.json')) as f:
            meta = json.load(f)
        offs, labs = meta['offsets'], meta['labels']
        for i in range(len(offs) - 1):
            items.append((f'{stem}_{i:04d}', stem, offs[i], offs[i + 1], labs[i]))
    return items


@torch.no_grad()
def validate(model, dirpath, device, score_thr=0.10, iou_thr=0.25,
             verbose=False, moto_from_bike=0.0, limit=0):
    model.eval()
    items = _shard_items(dirpath)
    if items is not None:
        return _validate_shard(model, dirpath, items, device, score_thr, iou_thr,
                               verbose, moto_from_bike, limit)
    gt = load_gt(dirpath)
    if limit and len(gt) > limit:
        # 均匀采样而不是取前 N 帧：M4 验证集跨 5000 多帧，只取开头会只看到
        # 一段场景，mAP 的方差大到没法用来选模型。
        fids = list(gt)
        idx = np.linspace(0, len(fids) - 1, limit).astype(int)
        keep = [fids[i] for i in sorted(set(idx.tolist()))]
        gt = {k: gt[k] for k in keep}
    preds = {}
    t0 = time.time()
    nbox = 0
    for fid in gt:
        pts = np.fromfile(os.path.join(dirpath, fid + '.bin'),
                          dtype=np.float32).reshape(-1, 3)
        b, s, c = M.detect(model, pts, device, score_thr, iou_thr,
                           moto_from_bike=moto_from_bike)
        preds[fid] = to_eval_format(b, s, c)
        nbox += len(b)
    res = evaluate_3d_detection(preds, gt, C.EVAL_CLASS_NAMES,
                                iou_thresholds=(0.5,), iou_type='3d')
    res['_secs'] = time.time() - t0
    res['_boxes_per_frame'] = nbox / max(len(gt), 1)
    if verbose:
        print('    每类 AP@0.5:')
        for name, v in res['per_class'].items():
            ap = v['AP_0.5_R40']
            print(f'      {name:<12} AP {ap:>6.3f}  GT {v["n_gt"]:>5}  预测 {v["n_pred"]:>5}')
    return res


@torch.no_grad()
def _validate_shard(model, dirpath, items, device, score_thr, iou_thr,
                    verbose, moto_from_bike, limit):
    """验证 shard 格式的合成集（.npy + _meta.json）。"""
    if limit and len(items) > limit:
        idx = np.linspace(0, len(items) - 1, limit).astype(int)
        items = [items[i] for i in sorted(set(idx.tolist()))]
    cache = {}
    preds, gt = {}, {}
    t0 = time.time()
    nbox = 0
    for fid, stem, a, b, labs in items:
        if stem not in cache:
            cache.clear()
            cache[stem] = np.load(os.path.join(dirpath, stem + '.npy'), mmap_mode='r')
        pts = np.asarray(cache[stem][a:b], dtype=np.float32)
        bb, ss, cc = M.detect(model, pts, device, score_thr, iou_thr,
                              moto_from_bike=moto_from_bike)
        preds[fid] = to_eval_format(bb, ss, cc)
        gt[fid] = [{'box': [float(v) for v in r[1:8]],
                    'class': C.CLASS_NAMES[int(r[0])]} for r in labs]
        nbox += len(bb)
    res = evaluate_3d_detection(preds, gt, C.EVAL_CLASS_NAMES,
                                iou_thresholds=(0.5,), iou_type='3d')
    res['_secs'] = time.time() - t0
    res['_boxes_per_frame'] = nbox / max(len(gt), 1)
    if verbose:
        print('    每类 AP@0.5:')
        for name, v in res['per_class'].items():
            ap = v['AP_0.5_R40']
            print(f'      {name:<12} AP {ap:>6.3f}  GT {v["n_gt"]:>5}  预测 {v["n_pred"]:>5}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--val-every', type=int, default=2)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--moto-from-bike', type=float, default=0.0)
    ap.add_argument('--cls-balance', type=int, default=0,
                    help='1=按类别重加权正样本（实测会压垮 Car，默认关）')
    ap.add_argument('--real-repeat', type=int, default=1,
                    help='真实帧在训练集里重复几次（155 帧只占 3.7 个百分点，微调阶段调大）')
    ap.add_argument('--init', default=None, help='从指定权重热启动（只加载模型）')
    ap.add_argument('--train-dirs', default=None,
                    help='逗号分隔的训练目录，默认 "合成 + 155 帧真实"。'
                         'M4 管线用 --train-dirs data/train/m4_train')
    ap.add_argument('--val-limit', type=int, default=0,
                    help='验证帧数上限（均匀采样），0=全部。M4 验证集 1246 帧全跑要 16 分钟')
    ap.add_argument('--val-dir', default=None,
                    help='验证目录，默认 data/train/val_real（31 帧）。'
                         'M4 管线用 --val-dir data/train/m4_val')
    a = ap.parse_args()

    os.makedirs(C.CKPT_DIR, exist_ok=True)
    os.makedirs(C.OUT_DIR, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cudnn autotune 在"每个 batch 的 pillar 数都不一样"时收益很小，却会反复
    # 重挑算法；实测开着会在 epoch 3 左右抛 `CUDA error: an illegal instruction`
    # （cudnn 在某个 shape 上选到了坏 kernel）。关掉后训练稳定。
    torch.backends.cudnn.benchmark = False

    if a.train_dirs:
        dirs = [d.strip() for d in a.train_dirs.split(',') if d.strip()]
    else:
        dirs = [C.TRAIN_SYNTH, C.TRAIN_REAL]
    for d in dirs:
        if not os.path.isdir(d):
            raise SystemExit(f'训练目录不存在：{d}')
    val_dir = a.val_dir or C.VAL_REAL
    if not os.path.isdir(val_dir):
        raise SystemExit(f'验证目录不存在：{val_dir}')
    print('训练目录: ' + '  '.join(dirs))
    print('验证目录: ' + val_dir)

    ds, dl = make_loader(dirs, True, a.bs, a.workers, True)
    if a.real_repeat > 1:
        real = [it for it in ds.items if it[0] < 0]
        ds.items = ds.items + real * (a.real_repeat - 1)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                        collate_fn=D.collate, pin_memory=True,
                        persistent_workers=(a.workers > 0), drop_last=True)
    n_syn = sum(1 for k, _ in ds.items if k >= 0)
    print(f'训练帧 {len(ds)}（合成 {n_syn}，真实 {len(ds)-n_syn}），'
          f'{len(dl)} iter/epoch')

    model = M.PointPillars().to(dev)
    if a.init:
        st0 = torch.load(a.init, map_location=dev, weights_only=False)
        model.load_state_dict(st0['model'])
        print(f'热启动自 {a.init}（epoch {st0.get("epoch")}, '
              f'val mAP {st0.get("best", -1):.4f}）')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(dev.type == 'cuda'))
    steps = a.epochs * len(dl)
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(steps - warm, 1))))

    best = -1.0
    start_ep = 0
    ck = os.path.join(C.CKPT_DIR, f'pp_{a.tag}.pt')
    ck_best = os.path.join(C.CKPT_DIR, f'pp_{a.tag}_best.pt')
    if a.resume and os.path.isfile(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start_ep = st['epoch'] + 1
        best = st.get('best', -1.0)
        print(f'从 {ck} 恢复，epoch {start_ep}，best mAP {best:.4f}')

    for ep in range(start_ep, a.epochs):
        model.train()
        t0 = time.time()
        run_c = run_r = 0.0
        nb = 0
        for it, batch in enumerate(dl):
            feat = batch['feat'].to(dev, non_blocking=True)
            coords = batch['coords'].to(dev, non_blocking=True)
            npts = batch['npts'].to(dev, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(dev.type == 'cuda')):
                cls, reg = model(feat, coords, npts)
                lc, lr_ = M.compute_loss(
                    cls, reg, batch['pos_idx'],
                    batch['reg_t'], batch['pos_cls'],
                    cls_balance=bool(a.cls_balance),
                    heat_list=(batch.get('heat')
                               if os.environ.get('PP_HEAT', '1') == '1' else None))
                loss = lc + 2.0 * lr_
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_c += lc.item()
            run_r += lr_.item()
            nb += 1
            if (it + 1) % 50 == 0:
                print(f'  ep{ep} {it+1}/{len(dl)}  cls {run_c/nb:.3f}  '
                      f'reg {run_r/nb:.3f}  lr {sched.get_last_lr()[0]:.2e}  '
                      f'{time.time()-t0:.0f}s', flush=True)
        print(f'ep{ep} 完成  cls {run_c/max(nb,1):.4f}  reg {run_r/max(nb,1):.4f}  '
              f'{time.time()-t0:.0f}s', flush=True)

        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            r = validate(model, val_dir, dev,
                         moto_from_bike=a.moto_from_bike, limit=a.val_limit)
            m = r['mAP_0.5_R40']
            print(f'  [验证 M4 真实帧] mAP@3D0.5 = {m:.4f}  '
                  f'({r["_secs"]:.0f}s, {r["_boxes_per_frame"]:.0f} 框/帧)', flush=True)
            if m > best:
                best = m
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                            'sched': sched.state_dict(), 'epoch': ep,
                            'best': best, 'map': m}, ck_best)
                print(f'  -> 新的最好结果，已存 {ck_best}', flush=True)
                with open(os.path.join(C.OUT_DIR, f'val_{a.tag}.json'), 'w') as f:
                    json.dump(r, f, indent=1, default=float)
            # 每个验证点单独留一份，便于训练后按真实帧 mAP 挑 epoch（best 判据
            # 只是单一阈值下的 mAP，不同 score_thr 下的最优 epoch 未必相同）。
            torch.save({'model': model.state_dict(), 'epoch': ep, 'best': best,
                        'map': m}, os.path.join(C.CKPT_DIR,
                                                f'pp_{a.tag}_ep{ep:02d}.pt'))
        else:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'epoch': ep, 'best': best}, ck)

    print(f'训练结束，最好 mAP@3D0.5 = {best:.4f}')
    r = validate(model, val_dir, dev, verbose=True, limit=a.val_limit)
    print(f'最终 mAP@3D0.5 = {r["mAP_0.5_R40"]:.4f}')


if __name__ == '__main__':
    main()
