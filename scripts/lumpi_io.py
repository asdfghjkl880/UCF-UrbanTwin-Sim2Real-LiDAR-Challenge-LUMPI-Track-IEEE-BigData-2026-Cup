#!/usr/bin/env python3
"""LUMPI 数据读写小工具（纯标准库，无需 numpy）。"""
import array
import os
import struct

BOX = (-40.0, -40.0, -5.0, 40.0, 40.0, 5.0)

_PLY_T = {"float": ("f", 4), "double": ("d", 8), "uchar": ("B", 1), "char": ("b", 1),
          "short": ("h", 2), "ushort": ("H", 2), "int": ("i", 4), "uint": ("I", 4)}


def read_ply(path):
    """返回 (header_dict, 每点属性 dict 的列表)。适合中小文件。"""
    with open(path, "rb") as f:
        lines, props, n, fmt = [], [], 0, None
        while True:
            ln = f.readline().decode("ascii", "replace").strip()
            lines.append(ln)
            if ln.startswith("format"):
                fmt = ln.split()[1]
            elif ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property"):
                p = ln.split()
                props.append((p[-1], p[1]))
            elif ln == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: 仅支持 binary_little_endian，实际 {fmt}")
        sizes = [_PLY_T[t][1] for _, t in props]
        fmtstr = "<" + "".join(_PLY_T[t][0] for _, t in props)
        stride = sum(sizes)
        raw = f.read(n * stride)

    names = [p[0] for p in props]
    out = {k: [] for k in names}
    for rec in struct.iter_unpack(fmtstr, raw):
        for k, v in zip(names, rec):
            out[k].append(v)
    return {"n": n, "props": names, "fmt": fmtstr, "stride": stride}, out


def read_ply_xyz(path):
    """只取 xyz，返回 [(x,y,z), ...]。走快速路径。"""
    _, d = read_ply(path)
    return list(zip(d["x"], d["y"], d["z"]))


def read_bin_xyz(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // 4)
    n = len(a) // 3
    return list(zip(a[0:3 * n:3], a[1:3 * n:3], a[2:3 * n:3]))


def write_bin_xyz(path, pts):
    a = array.array("f")
    for p in pts:
        a.extend((float(p[0]), float(p[1]), float(p[2])))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        a.tofile(f)


def crop_box(pts, box=BOX):
    x0, y0, z0, x1, y1, z1 = box
    return [p for p in pts if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1]


def subsample(pts, k, seed=1234):
    """确定性均匀子采样（不依赖 numpy）。"""
    import random
    if len(pts) <= k:
        return list(pts)
    rng = random.Random(seed)
    return [pts[i] for i in sorted(rng.sample(range(len(pts)), k))]
