#!/usr/bin/env python3
"""把中间产物送进 Windows 回收站（不是永久删除）。

用 Shell 的 SHFileOperationW + FOF_ALLOWUNDO，等价于在资源管理器里按 Delete，
文件进回收站，可随时还原。

用法：
    python scripts/to_recycle.py            # 只列清单
    python scripts/to_recycle.py --apply    # 送进回收站
"""
import ctypes
import os
import sys
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 保留：提交权重
KEEP_CKPT = {'pp_s7_ep07.pt', 'pp_s7_best.pt'}

# 早期提交帧目录（已被 data/synthetic_v7 取代）
OLD_SYNTH_DIRS = ['synthetic', 'synthetic_affine', 'synthetic_probe',
                  'synthetic_probe2', 'synthetic_v1', 'synthetic_v5',
                  'synthetic_v6', 'synth_probe3',
                  'sp_a', 'sp_b', 'sp_c', 'sp_d', 'sp_e']

FO_DELETE = 3
FOF_SILENT = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO = 0x0040
FOF_NOCONFIRMMKDIR = 0x0200
FOF_NOERRORUI = 0x0400


class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ('hwnd', wintypes.HWND),
        ('wFunc', wintypes.UINT),
        ('pFrom', wintypes.LPCWSTR),
        ('pTo', wintypes.LPCWSTR),
        ('fFlags', ctypes.c_uint16),
        ('fAnyOperationsAborted', wintypes.BOOL),
        ('hNameMappings', ctypes.c_void_p),
        ('lpszProgressTitle', wintypes.LPCWSTR),
    ]


def _one(path):
    """单个路径送回收站。逐条调用：多路径的 pFrom 列表在 ctypes 下会被
    Python 的宽字符串转换在第一个 \\0 处截断，实测返回 0x2（找不到文件）。"""
    fn = ctypes.windll.shell32.SHFileOperationW
    fn.argtypes = [ctypes.POINTER(SHFILEOPSTRUCTW)]
    fn.restype = ctypes.c_int
    op = SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = FO_DELETE
    op.pFrom = os.path.abspath(path) + '\0\0'
    op.pTo = None
    op.fFlags = (FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT |
                 FOF_NOERRORUI | FOF_NOCONFIRMMKDIR)
    op.fAnyOperationsAborted = False
    op.hNameMappings = None
    op.lpszProgressTitle = None
    return fn(ctypes.byref(op))


def to_recycle(paths):
    """把一组路径送进回收站，返回 (成功数, 首个错误码)。"""
    ok, first_err = 0, 0
    for p in paths:
        rc = _one(p)
        if rc == 0:
            ok += 1
        elif not first_err:
            first_err = rc
    return ok, first_err


def size_of(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


def collect():
    root = os.path.abspath(ROOT)
    out = []

    # 1) data/out 里的中间 zip（保留 submission.zip）
    d = os.path.join(root, 'data', 'out')
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.zip') and f != 'submission.zip':
                out.append(os.path.join(d, f))

    # 2) 旧权重
    d = os.path.join(root, 'data', 'train', 'ckpt')
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.pt') and f not in KEEP_CKPT:
                out.append(os.path.join(d, f))

    # 3) 被否掉方案的产物
    for rel in ('data/train/depthmap.npz', 'data/train/out'):
        p = os.path.join(root, rel.replace('/', os.sep))
        if os.path.exists(p):
            out.append(p)

    # 4) 早期提交帧目录
    for name in OLD_SYNTH_DIRS:
        p = os.path.join(root, 'data', name)
        if os.path.exists(p):
            out.append(p)

    # 5) __pycache__
    for r, ds, _ in os.walk(root):
        for x in list(ds):
            if x == '__pycache__':
                out.append(os.path.join(r, x))
                ds.remove(x)
    return out


def main():
    items = collect()
    items = [(p, size_of(p)) for p in items]
    items.sort(key=lambda x: -x[1])
    tot = sum(s for _, s in items)

    root = os.path.abspath(ROOT)
    print(f'{"体积":>9}  路径')
    print('-' * 64)
    for p, s in items:
        print(f'{s/1e6:>8.1f}M  {os.path.relpath(p, root)}')
    print('-' * 64)
    print(f'共 {len(items)} 项，合计 {tot/1e9:.2f} GB')

    if '--apply' not in sys.argv:
        print()
        print('dry-run：加 --apply 送进回收站')
        return 0

    n, rc = to_recycle([p for p, _ in items])
    if rc:
        print(f'SHFileOperation 返回错误码 0x{rc:04X}')
        return 1
    print(f'已送进回收站 {n} 项，约 {tot/1e9:.2f} GB。'
          f'（空间在清空回收站后才真正释放）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
