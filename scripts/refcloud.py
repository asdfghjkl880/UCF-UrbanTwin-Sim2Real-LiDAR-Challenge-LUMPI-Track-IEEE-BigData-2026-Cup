#!/usr/bin/env python3
"""
静态地图（LUMPI_reference_cloud）与真实帧 PLY 的统一读取。

地图：EPSG:25832 的 UTM 坐标，double x/y
    property double x, y / float z / double time / short reflectance / int run-id
真实帧：传感器局部坐标，float x/y/z + time/id/intensity/ray/azimuth/distance

本模块按 PLY 头部自动构造 numpy dtype，不写死字段偏移。
"""
import os
import numpy as np

from paths import REF_CLOUD, PREVIEW_CLOUD

_PLY_TYPES = {
    'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1',
    'short': 'i2', 'ushort': 'u2', 'int16': 'i2', 'uint16': 'u2',
    'int': 'i4', 'uint': 'u4', 'int32': 'i4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
}


def parse_ply_header(path):
    """返回 (元素个数, numpy dtype, 数据起始偏移)。只支持单一 vertex 元素的二进制 PLY。"""
    with open(path, 'rb') as f:
        head = f.read(8000)
    end = head.find(b'end_header')
    if end < 0:
        raise ValueError(f'{path} 不是合法 PLY')
    txt = head[:end].decode('ascii', 'ignore')
    off = end + len(b'end_header') + 1

    n = 0
    props = []
    in_vertex = False
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith('format') and 'binary_little_endian' not in s:
            raise ValueError(f'只支持 binary_little_endian：{path}')
        if s.startswith('element'):
            parts = s.split()
            in_vertex = parts[1] == 'vertex'
            if in_vertex:
                n = int(parts[2])
        elif s.startswith('property') and in_vertex:
            parts = s.split()
            if parts[1] == 'list':
                raise ValueError('不支持 list 属性')
            props.append((parts[2], '<' + _PLY_TYPES[parts[1]]))
    return n, np.dtype(props), off


def read_ply(path, limit=None):
    """读 PLY，返回结构化数组。"""
    n, dt, off = parse_ply_header(path)
    count = n if limit is None else min(n, limit)
    return np.fromfile(path, dtype=dt, count=count, offset=off)


def read_xyz(path, limit=None):
    """读 PLY，返回 (N,3) float64 的 xyz。"""
    a = read_ply(path, limit)
    return np.stack([a['x'].astype(np.float64),
                     a['y'].astype(np.float64),
                     a['z'].astype(np.float64)], axis=1)


def read_ref_cloud(path=None, preview=True, limit=None):
    """读静态地图点云（默认用 1/16 降采样的预览版，4M 点，1.6 秒）。"""
    if path is None:
        path = PREVIEW_CLOUD if preview else REF_CLOUD
    return read_xyz(path, limit)


def ground_z(pts, res=2.0):
    """估计地面高程：2m 网格内取 z 的 10% 分位数，再取中位数。"""
    if len(pts) == 0:
        return 0.0
    gx = np.floor(pts[:, 0] / res).astype(np.int64)
    gy = np.floor(pts[:, 1] / res).astype(np.int64)
    key = gx * 1000003 + gy
    order = np.argsort(key, kind='stable')
    key_s, z_s = key[order], pts[order, 2]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    lows, start = [], 0
    for b in list(bounds) + [len(key_s)]:
        seg = z_s[start:b]
        if len(seg) >= 20:
            lows.append(np.percentile(seg, 10))
        start = b
    return float(np.median(lows)) if lows else float(np.percentile(pts[:, 2], 10))


if __name__ == '__main__':
    p = read_ref_cloud()
    lo, hi = p.min(axis=0), p.max(axis=0)
    print(f'地图点数 {len(p):,}')
    print(f'  x {lo[0]:.1f} ~ {hi[0]:.1f}  (跨度 {hi[0]-lo[0]:.1f} m)')
    print(f'  y {lo[1]:.1f} ~ {hi[1]:.1f}  (跨度 {hi[1]-lo[1]:.1f} m)')
    print(f'  z {lo[2]:.1f} ~ {hi[2]:.1f}')
    print(f'  地面高程 {ground_z(p):.2f} m')
