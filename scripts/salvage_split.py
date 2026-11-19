#!/usr/bin/env python
"""从 split-zip 的最后一个分卷中抢救尾部完整文件。

背景：LUMPI 的 lidar.zip 按固定 4.375 GiB 切成 .001/.002/... 分卷。
只下载到最后一卷时，虽然前面若干卷缺失，但最后一卷的尾部
可能仍完整包含若干 .ply（其 local header 与压缩数据全部落在本卷内）。
最后卷带有完整中央目录（含 zip64 extra field 的真实 64 位偏移），
据此可精确判断哪些文件是完整可提取的。

全程只读官方数据，不涉及任何标签。
"""
import os
import struct
import zipfile
import zlib
import argparse

VOL = 4697620480  # 4.375 GiB，7-Zip 默认卷大小


def parse_cd(path):
    """解析最后卷尾部的中央目录，返回 (entries, cd_global_end)。

    entry: dict(name, csize, usize, lho, method)
    lho 为拼接后全流的全局偏移。
    """
    sz = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, sz - 16 * 1024 * 1024))
        buf = f.read()
        base = sz - len(buf)

    ents = []
    i = 0
    while True:
        j = buf.find(b"PK\x01\x02", i)
        if j < 0:
            break
        i = j + 4
        try:
            method = struct.unpack("<H", buf[j + 10 : j + 12])[0]
            csize = struct.unpack("<I", buf[j + 20 : j + 24])[0]
            usize = struct.unpack("<I", buf[j + 24 : j + 28])[0]
            nlen = struct.unpack("<H", buf[j + 28 : j + 30])[0]
            elen = struct.unpack("<H", buf[j + 30 : j + 32])[0]
            lho = struct.unpack("<I", buf[j + 42 : j + 46])[0]
            name = buf[j + 46 : j + 46 + nlen].decode("utf-8", "replace")
        except Exception:
            continue
        # 解析 zip64 extra field（tag 0x0001），字段顺序：usize, csize, lho
        extra = buf[j + 46 + nlen : j + 46 + nlen + elen]
        k = 0
        while k + 4 <= len(extra):
            tag, ln = struct.unpack("<HH", extra[k : k + 4])
            body = extra[k + 4 : k + 4 + ln]
            if tag == 1:
                p = 0
                if usize == 0xFFFFFFFF and p + 8 <= len(body):
                    usize = struct.unpack("<Q", body[p : p + 8])[0]
                    p += 8
                if csize == 0xFFFFFFFF and p + 8 <= len(body):
                    csize = struct.unpack("<Q", body[p : p + 8])[0]
                    p += 8
                if lho == 0xFFFFFFFF and p + 8 <= len(body):
                    lho = struct.unpack("<Q", body[p : p + 8])[0]
                    p += 8
            k += 4 + ln
        ents.append(dict(name=name, csize=csize, usize=usize, lho=lho, method=method))
    return ents


def salvage(vol_path, out_dir, nvol, limit=None, dry=False):
    """提取完整落在最后卷内的文件。nvol = 该 zip 的总卷数。"""
    g0 = (nvol - 1) * VOL  # 最后卷在拼接流中的起始全局偏移
    vsz = os.path.getsize(vol_path)
    g_end = g0 + vsz
    ents = parse_cd(vol_path)
    print(f"{os.path.basename(vol_path)}: CD {len(ents)} 条, 最后卷全局区间 [{g0}, {g_end})")

    cand = [
        e
        for e in ents
        if e["name"].endswith(".ply") and g0 <= e["lho"] < g_end
    ]
    print(f"  落在最后卷内的条目: {len(cand)}")
    if limit:
        cand = cand[:limit]

    os.makedirs(out_dir, exist_ok=True)
    ok = 0
    tot = 0
    with open(vol_path, "rb") as f:
        for e in cand:
            off = e["lho"] - g0
            if off < 0 or off + 30 > vsz:
                continue
            f.seek(off)
            lh = f.read(30)
            if lh[:4] != b"PK\x03\x04":
                continue  # 本地头不在本卷，跳过
            nlen = struct.unpack("<H", lh[26:28])[0]
            elen = struct.unpack("<H", lh[28:30])[0]
            dstart = off + 30 + nlen + elen
            if dstart + e["csize"] > vsz:
                continue  # 数据跨卷，跳过
            f.seek(dstart)
            raw = f.read(e["csize"])
            if e["method"] == 8:
                try:
                    data = zlib.decompress(raw, -15)
                except Exception:
                    continue
            elif e["method"] == 0:
                data = raw
            else:
                continue
            if len(data) != e["usize"]:
                continue
            if not dry:
                dst = os.path.join(out_dir, os.path.basename(e["name"]))
                with open(dst, "wb") as g:
                    g.write(data)
            ok += 1
            tot += len(data)
    print(f"  抢救成功 {ok} 个文件, 解压后 {tot/2**20:.1f} MB")
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nvol", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    salvage(a.vol, a.out, a.nvol, a.limit, a.dry)


if __name__ == "__main__":
    main()
