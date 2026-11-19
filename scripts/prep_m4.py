#!/usr/bin/env python3
"""
把下载的 Measurement4 完整点云转成 PointPillars 训练格式。

这是"真实数据训练"这一阶段的核心脚本。分两步：

  1) extract —— 把 lidar.zip.001 / .002 / ... 分卷拼成完整 zip 再解压
  2) convert —— 逐帧裁到 ±40m/±5m、子采样到 50,000 点，按时间切分训练/验证

合规要点（重要，别改）：
  * forbidden_frames.txt 里 100 个帧号是明令禁止的，必须整段避开。
    这 100 个 ID 稀疏分布在 6497~8109 之间，而相邻帧只差 0.1 秒、几乎重复，
    所以默认**整段排除 6497~8109**（1613 帧），而不是只删那 100 个 ID。
    用 --only-listed 可以改成只排除列表里的 100 个（不建议）。
  * 帧 0~30 是本地验证集 val_real 的来源，默认从训练集里排除，
    否则本地 mAP 会虚高。

产出：
  data/train/m4_train/<fid>.bin + <fid>.json
  data/train/m4_val/<fid>.bin + <fid>.json

用法：
    # 先看会转换多少帧，不动磁盘
    python scripts/prep_m4.py convert --dry-run
    # 正式跑
    python scripts/prep_m4.py all --src "D:/data/Microsoft Edge"
    # 只转换前 200 帧做冒烟测试
    python scripts/prep_m4.py convert --limit 200
"""
import argparse
import csv
import json
import os
import shutil
import sys
import zipfile
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, LABEL_DIR, LUMPI  # noqa: E402
import pp_config as C  # noqa: E402

N_POINTS = 50000
FPS = 10.0

# 禁用区间（含端点）。100 个明令禁止的帧号稀疏分布在其中。
FORBID_LO, FORBID_HI = 6497, 8109
VAL_REAL_MAX_ID = 30          # 帧 0~30 已作本地验证集
# M4 实际帧数：分卷 zip 的 EOCD 记录 8201 个条目、末卷含 lidar/008199.ply，
# 所以是 0~8199 共 8200 帧（早前按 8120 估算是错的）。多算不存在的帧无害，
# _convert_one 找不到 .ply 会返回 None。
N_FRAMES_MAX = 8200

RAW_DIR = os.path.join(LUMPI, 'Measurement4', 'lidar')
ZIP_DIR = os.path.join(LUMPI, 'Measurement4')
OUT_TRAIN = os.path.join(DATA, 'train', 'm4_train')
OUT_VAL = os.path.join(DATA, 'train', 'm4_val')
LABEL_CSV = os.path.join(LABEL_DIR, 'Measurement4', 'Label.csv')

PLY_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('time', '<u4'),
    ('id', 'u1'), ('intensity', 'u1'), ('ray', 'u1'),
    ('azimuth', '<u2'), ('distance', '<u2'),
])


# ---------------------------------------------------------------- 帧号 / 标注
def listed_forbidden():
    p = os.path.join(os.path.dirname(DATA), 'lumpi_starting_kit',
                     'forbidden_frames.txt')
    out = set()
    if not os.path.exists(p):
        return out
    for line in open(p, encoding='utf-8', errors='replace'):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        try:
            out.add(int(s))
        except ValueError:
            pass
    return out


def legal_ids(only_listed=False):
    """返回可用的帧号列表（已排除禁用段与本地验证帧）。"""
    forbid = listed_forbidden()
    if only_listed:
        bad = forbid
        test = lambda f: f in bad
    else:
        test = lambda f: FORBID_LO <= f <= FORBID_HI

    ids = []
    for f in range(0, N_FRAMES_MAX):
        if f <= VAL_REAL_MAX_ID:
            continue
        if test(f):
            continue
        ids.append(f)
    return ids


def load_labels(path=LABEL_CSV):
    """{frame_id: [[cls, cx, cy, cz, L, W, H, hd], ...]}，列位置见 csv.reader。"""
    out = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 16:
                continue
            try:
                fid = int(round(float(row[0]) * FPS))
                cls = int(float(row[7]))
                cx, cy, cz = float(row[9]), float(row[10]), float(row[11])
                L, W, H = float(row[12]), float(row[13]), float(row[14])
                hd = float(row[15])
            except (ValueError, IndexError):
                continue
            if cls > 6:                       # class 7 测试时段 0 实例，丢弃
                continue
            if not (L > 0.1 and W > 0.1 and H > 0.1):
                continue
            if not np.isfinite([cx, cy, cz, L, W, H, hd]).all():
                continue
            out.setdefault(fid, []).append([cls, cx, cy, cz, L, W, H, hd])
    return out


# ---------------------------------------------------------------- 解压
# 浏览器下载分卷时经常把文件名弄丢（变成 "未确认 219547.crdownload" 或
# "未确认 219547"），而 7-Zip 的分卷是**纯字节切片**，卷内没有 disk number
# 标记，所以无法靠文件头判断顺序。唯一可靠的办法是扫每卷里出现的
# "lidar/XXXXXX.ply" 文件名：卷内最小文件名越小，该卷越靠前。
_PLY_PAT = b'PK\x03\x04'


def scan_volume_names(path, max_hits=400):
    """扫一个分卷里出现的 .ply 文件名（只读 local file header，不依赖中央目录）。"""
    hits = []
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 26)          # 64 MB
            if not chunk:
                break
            i = 0
            while True:
                j = chunk.find(_PLY_PAT, i)
                if j < 0 or j + 30 > len(chunk):
                    break
                nl = int.from_bytes(chunk[j + 26:j + 28], 'little')
                if 5 < nl < 200 and j + 30 + nl <= len(chunk):
                    try:
                        s = chunk[j + 30:j + 30 + nl].decode('utf-8')
                    except UnicodeDecodeError:
                        s = ''
                    if s.endswith('.ply'):
                        hits.append(s)
                        if len(hits) >= max_hits:
                            return hits
                i = j + 4
    return hits


def collect_volumes(src):
    """收集分卷并按卷内最小 .ply 文件名排序。

    兼容两种命名：7-Zip 的 lidar.zip.001/.002 和浏览器丢名字的
    "未确认 219547"。跳过 .crdownload（没下完）。
    """
    cand = []
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if not os.path.isfile(p) or f.endswith('.crdownload'):
            continue
        if f.startswith('_lidar_merged'):
            continue
        sz = os.path.getsize(p)
        if sz < 1 << 20:                      # 太小的不是分卷
            continue
        cand.append(p)
    if not cand:
        return []

    # 有规范命名的直接按名字排，省掉扫描（每卷 4 GB 扫一遍要几十秒）
    named = [p for p in cand if os.path.basename(p).startswith('lidar.zip.')
             and os.path.basename(p)[-3:].isdigit()]
    if len(named) == len(cand):
        return sorted(named)

    print(f'分卷命名不规范，扫描 {len(cand)} 个卷内的文件名来推断顺序…')
    keyed = []
    for p in cand:
        nm = scan_volume_names(p, max_hits=64)
        key = min(nm) if nm else '\uffff'     # 扫不到排最后（多半是末卷）
        keyed.append((key, p, len(nm)))
        print(f'  {os.path.basename(p)[:34]:>36}  {os.path.getsize(p)/2**30:6.2f} GB  '
              f'首帧 {key if key != chr(0xffff) else "?"}')
    keyed.sort(key=lambda t: t[0])
    return [t[1] for t in keyed]


def extract(src=None, dst=RAW_DIR):
    """把分卷拼成完整 zip 再解压到 dst。"""
    src = src or ZIP_DIR
    parts = collect_volumes(src)
    if not parts:
        raise SystemExit(f'{src} 里没找到 zip 分卷')

    total = sum(os.path.getsize(p) for p in parts)
    print(f'分卷 {len(parts)} 个，共 {total/1e9:.2f} GB')

    merged = os.path.join(src, '_lidar_merged.zip')
    if not os.path.exists(merged) or os.path.getsize(merged) != total:
        print(f'拼接 -> {merged}')
        with open(merged, 'wb') as w:
            for p in parts:
                with open(p, 'rb') as r:
                    shutil.copyfileobj(r, w, 1 << 22)
    else:
        print(f'已存在拼接结果，跳过：{merged}')

    os.makedirs(dst, exist_ok=True)
    print(f'解压 -> {dst}')
    n = 0
    with zipfile.ZipFile(merged) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith('.ply')]
        print(f'压缩包内 {len(members)} 个 .ply')
        for m in members:
            base = os.path.basename(m)
            tgt = os.path.join(dst, base)
            if os.path.exists(tgt) and os.path.getsize(tgt) > 0:
                continue
            with zf.open(m) as fsrc, open(tgt, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst, 1 << 22)
            n += 1
            if n % 500 == 0:
                print(f'  已解压 {n}', flush=True)
    print(f'解压完成，新增 {n} 个文件 -> {dst}')


# ---------------------------------------------------------------- 转换
def read_ply_xyz(path):
    with open(path, 'rb') as f:
        raw = f.read()
    i = raw.index(b'end_header\n') + len(b'end_header\n')
    a = np.frombuffer(raw[i:], dtype=PLY_DTYPE)
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float32)


def _convert_one(args):
    fid, out_dir = args
    p = os.path.join(RAW_DIR, f'{fid:06d}.ply')
    if not os.path.exists(p):
        return fid, None
    pts = read_ply_xyz(p)
    m = ((pts[:, 0] >= C.X_MIN) & (pts[:, 0] <= C.X_MAX) &
         (pts[:, 1] >= C.Y_MIN) & (pts[:, 1] <= C.Y_MAX) &
         (pts[:, 2] >= C.Z_MIN) & (pts[:, 2] <= C.Z_MAX))
    p2 = pts[m]
    if len(p2) > N_POINTS:
        rng = np.random.default_rng(fid)
        p2 = p2[rng.choice(len(p2), N_POINTS, replace=False)]
    if len(p2) < 1000:
        return fid, None
    p2.tofile(os.path.join(out_dir, f'{fid:06d}.bin'))
    return fid, len(p2)


def convert(limit=0, workers=4, dry_run=False, only_listed=False):
    ids = legal_ids(only_listed)
    labs = load_labels()
    present = set()
    if os.path.isdir(RAW_DIR):
        present = {int(os.path.splitext(f)[0])
                   for f in os.listdir(RAW_DIR) if f.endswith('.ply')}

    if present:
        # 必须有标注：M4 标注只到 8109，8110~8199 那 80 帧虽然有点云但没标签，
        # 转进来就是纯负样本，会把分类头教坏。
        todo = [f for f in ids if f in present and labs.get(f)]
        missing = len(ids) - len(todo)
    else:
        todo = [f for f in ids if labs.get(f)]
        missing = 0
    if limit:
        todo = todo[:limit]

    # 按时间切：块内最后 20% 作验证，保证验证集与训练集时间上不重叠。
    # 中间留 GAP 帧缓冲：相邻帧只差 0.1 秒、几乎重复，紧挨着切会让验证集
    # 和训练集共享同一段场景，mAP 虚高。
    cut = int(len(todo) * 0.8)
    gap = min(50, max(0, len(todo) - cut - 1))
    train_ids = todo[:cut]
    val_ids = todo[cut + gap:]
    val_set = set(val_ids)

    print(f'合法帧 {len(ids)}  磁盘上有 {len(present & set(ids))}  '
          f'有标注且可转 {len(todo)}'
          f'{"  （缺 %d）" % missing if missing else ""}')
    print(f'  训练 {len(train_ids)} 帧（{train_ids[0]}~{train_ids[-1]}）  '
          f'验证 {len(val_ids)} 帧（{val_ids[0] if val_ids else "-"}~'
          f'{val_ids[-1] if val_ids else "-"}）  间隔 {gap} 帧')
    if dry_run:
        return

    for d in (OUT_TRAIN, OUT_VAL):
        os.makedirs(d, exist_ok=True)

    # 只转 train + val，**gap 帧必须排除**：它夹在两者之间，转进训练集就等于
    # 验证集紧邻训练集，泄漏回来的 mAP 会虚高。
    jobs = ([(f, OUT_TRAIN) for f in train_ids] +
            [(f, OUT_VAL) for f in val_ids])
    done = 0
    with Pool(workers) as pool:
        for fid, n in pool.imap_unordered(_convert_one, jobs, chunksize=16):
            if n is None:
                continue
            out_dir = OUT_VAL if fid in val_set else OUT_TRAIN
            with open(os.path.join(out_dir, f'{fid:06d}.json'), 'w') as f:
                json.dump(labs.get(fid, []), f)
            done += 1
            if done % 250 == 0:
                print(f'  转换 {done}/{len(jobs)}', flush=True)
    print(f'完成 {done} 帧 -> {OUT_TRAIN} / {OUT_VAL}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['extract', 'convert', 'all', 'stats'])
    ap.add_argument('--src', default=None, help='分卷所在目录')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only-listed', action='store_true',
                    help='只排除 forbidden_frames.txt 的 100 个 ID（不建议）')
    a = ap.parse_args()

    if a.cmd == 'stats':
        ids = legal_ids(a.only_listed)
        present = {int(os.path.splitext(f)[0])
                   for f in os.listdir(RAW_DIR) if f.endswith('.ply')} \
            if os.path.isdir(RAW_DIR) else set()
        labs = load_labels()
        print(f'合法帧 {len(ids)}（{ids[0]}~{ids[-1]}）')
        print(f'磁盘上已有 {len(present & set(ids))} 帧')
        print(f'标注覆盖 {len(set(labs) & set(ids))} 帧，'
              f'共 {sum(len(labs.get(i, [])) for i in ids)} 个框')
        return

    if a.cmd in ('extract', 'all'):
        extract(a.src)
    if a.cmd in ('convert', 'all'):
        convert(a.limit, a.workers, a.dry_run, a.only_listed)


if __name__ == '__main__':
    main()
