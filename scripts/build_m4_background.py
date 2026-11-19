"""
构建 Measurement4 的"静态背景距离图"——渲染合成帧的地基。

思路：物体只会让测距变**近**，所以同一光束在 31 帧里取最大值就是背景。
官方 background/*.npy 也是背景，但存在两个问题：
  * 部分扇区是空值（无回波），而真实帧在那里有回波（session 34 的 60~75°）
  * 部分扇区拍摄时刻不同，场景已经变了（session 34 的 105~180°、session 36 整体）

所以把两者融合：优先用帧最大值；帧没有回波的地方用官方图兜底；官方图与帧
最大值差得离谱的地方（>3 m）判定为"官方图过期"，丢弃。

输出 data/LUMPI/bg_m4_{sid}.npy，float32，单位米，0 表示无回波。
"""

import os

import numpy as np

from m4_sensor import IMG_SHAPE, K_M, LIDAR_DIR, M4Sensors, load_frame
from paths import M4_BG_DIR

STALE_TOL = 3.0     # 官方图与帧最大值差超过此值 -> 判为过期


def build(sid, n_frames=31, sensor=None):
    s = sensor or M4Sensors()
    nr, nc = IMG_SHAPE[sid]
    mx = np.zeros((nr, nc), np.float32)
    cnt = np.zeros((nr, nc), np.int16)

    for f in range(n_frames):
        p = os.path.join(LIDAR_DIR, f"{f:06d}.ply")
        if not os.path.exists(p):
            continue
        a, _ = load_frame(p)
        m = a["id"] == sid
        if m.sum() < 50:
            continue
        ray = a["ray"][m].astype(int)
        c = s.col_of(sid, s.azimuth_deg(a["azimuth"][m]))
        d = a["distance"][m].astype(np.float32) * K_M
        np.maximum.at(mx, (ray, c), d)
        np.add.at(cnt, (ray, c), 1)

    off = s.background_range(sid)
    src = np.zeros((nr, nc), np.uint8)          # 0=空 1=帧 2=官方 3=融合
    out = np.zeros((nr, nc), np.float32)

    has_f = mx > 0
    has_o = off > 0
    out[has_f] = mx[has_f]
    src[has_f] = 1

    both = has_f & has_o
    agree = both & (np.abs(mx - off) < STALE_TOL)

    # 只有官方图有 -> 直接采用
    only_o = ~has_f & has_o
    out[only_o] = off[only_o]
    src[only_o] = 2

    # 两边都有且一致 -> 取较远的（更可能是真背景）
    out[agree] = np.maximum(mx[agree], off[agree])
    src[agree] = 3

    stats = {
        "session": sid,
        "coverage": float(np.mean(out > 0)),
        "from_frames": float(np.mean(has_f)),
        "from_official": float(np.mean(only_o)),
        "agree": float(np.mean(agree)),
        "official_stale": float(np.mean(both & ~agree)),
    }
    return out, src, stats


def main():
    s = M4Sensors()
    os.makedirs(M4_BG_DIR, exist_ok=True)
    rows = []
    for sid in sorted(IMG_SHAPE):
        bg, src, st = build(sid, sensor=s)
        np.save(os.path.join(M4_BG_DIR, f"bg_m4_{sid}.npy"), bg)
        np.save(os.path.join(M4_BG_DIR, f"bgsrc_m4_{sid}.npy"), src)
        rows.append(st)

    print(f"{'session':>8} {'覆盖率':>8} {'帧来源':>8} {'官方兜底':>9} "
          f"{'一致':>8} {'官方过期':>9}")
    for r in rows:
        print(f"{r['session']:>8} {100*r['coverage']:7.1f}% "
              f"{100*r['from_frames']:7.1f}% {100*r['from_official']:8.1f}% "
              f"{100*r['agree']:7.1f}% {100*r['official_stale']:8.1f}%")

    # 验证：真实帧里"测得比背景近"的光束占比，应与 GT 框内点占比同量级
    print("\n验证：每帧'物体遮挡光束'占比 vs GT 框内点占比")
    a, _ = load_frame(os.path.join(LIDAR_DIR, "000000.ply"))
    for sid in sorted(IMG_SHAPE):
        bg = np.load(os.path.join(M4_BG_DIR, f"bg_m4_{sid}.npy"))
        m = a["id"] == sid
        if m.sum() < 200:
            continue
        ray = a["ray"][m].astype(int)
        c = s.col_of(sid, s.azimuth_deg(a["azimuth"][m]))
        d = a["distance"][m].astype(np.float32) * K_M
        b = bg[ray, c]
        ok = b > 0
        frac = float(np.mean(d[ok] < b[ok] - 0.5))
        print(f"  session {sid:>2}: 遮挡光束占 {100*frac:4.1f}%")


if __name__ == "__main__":
    main()
