"""
LUMPI Measurement4 传感器模型 —— 从真实帧反解出的完整光束几何。

这套解码是把官方数据里"看不出来"的编码关系全部解开的产物，核心结论：

  1. 一帧 .ply 是 5 台激光雷达融合的结果，`id` 字段直接就是 session id
     （34/35/36/37/38），不是设备号。
  2. `distance` 字段的单位是 0.005 m（1/200 m），满量程 20000 单位 = 100 m。
     依据：对 5 个 session 独立做 Kabsch 配准，拟合出的尺度全部是 0.005000。
  3. `ray` 是通道号，俯仰角 = **负的** meta.json 里 `angles[ray]`（度）。
     符号必须取反，否则残差从 0.018 m 涨到 4.4 m。
  4. `azimuth` 单位是 0.01°，`azimuth/100` 直接就是方位角（度），无需偏移。
     之前以为需要 90° 偏移，那是因为没做符号翻转时的补偿假象。
  5. 距离是**斜距**（slant range），不是水平距离。用水平距离模型残差 0.18 m，
     用斜距 0.018 m。
  6. 背景距离图 `background/{34..38}.npy` 是**同一套光束网格上的空场景距离图**，
     单位同样是 0.005 m，索引关系是 `[ray, round(azimuth_deg / (360/ncols))]`。
     session 37/38 还原到世界坐标后与实测点中位误差 **7 cm**。

已知缺陷（用的时候要绕开）：

  * session 35（device 0）的 `angles` 数组不是纯俯仰序——数值呈 4 个一组交错，
    直接索引残差 1.76 m。需要时用 `empirical_elevation()` 从数据反推。
  * session 36（device 2）`angles` 范围 -52°~+31° 明显异常，残差 0.024 m 但只有
    3115 个点，且背景图对该 session 偏差 44 m。基本可以不用。
  * session 34 的背景图在方位 45°~180° 扇区与当前场景不符（背景拍摄时刻不同），
    该扇区中位误差 2.4~11.2 m；其余扇区 0.12 m。用之前要按扇区打掩码。

用法：
    from m4_sensor import M4Sensors, load_frame
    s = M4Sensors()
    f = load_frame(path)                 # 结构化数组
    pts = f['xyz']                       # (N,3) 世界坐标
    dirs = s.beam_dirs(34)               # (64,2250,3) 每个光束的世界方向单位向量
    bg = s.background_world(34)          # (N,3) 背景图还原的世界点
"""

import json
import os

import numpy as np

from paths import LUMPI, M4_GEOM

K_M = 0.005          # distance 字段 / 背景图数值 -> 米
META_JSON = os.path.join(LUMPI, "test_data", "test_data", "meta.json")
BG_DIR = os.path.join(LUMPI, "Measurement4", "background", "background")
LIDAR_DIR = os.path.join(LUMPI, "test_data", "test_data", "Measurement4", "lidar")
GEOM_JSON = M4_GEOM

PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("time", "<u4"),
    ("id", "u1"), ("intensity", "u1"), ("ray", "u1"),
    ("azimuth", "<u2"), ("distance", "<u2"),
])

# 每个 session 的背景图尺寸；列数决定方位分辨率（360/ncols）
IMG_SHAPE = {34: (64, 2250), 35: (64, 2250), 36: (64, 2250),
             37: (16, 1800), 38: (16, 1800)}

# 已知有问题的 session，渲染时建议排除
BROKEN = {35: "angles 数组非俯仰序", 36: "angles 异常 + 背景图偏差 44m"}


def load_frame(path):
    """读一帧 .ply，返回带 `xyz` 视图的结构化数组。"""
    with open(path, "rb") as fh:
        raw = fh.read()
    i = raw.index(b"end_header\n") + len(b"end_header\n")
    a = np.frombuffer(raw[i:], dtype=PLY_DTYPE).copy()
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32)
    return a, xyz


def kabsch(L, P):
    """求 k, R, t 使 k * R @ L + t ≈ P（L、P 同尺度，单位米）。"""
    lc, pc = L.mean(0), P.mean(0)
    L0, P0 = L - lc, P - pc
    U, S, Vt = np.linalg.svd(L0.T @ P0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    k = S.sum() / (L0 ** 2).sum()
    return float(k), R, pc - k * (R @ lc)


class M4Sensors:
    """5 台雷达的几何参数、光束方向、背景距离图。"""

    def __init__(self, geom_path=GEOM_JSON):
        self.meta = json.load(open(META_JSON, encoding="utf-8"))
        self.sessions = sorted(int(k) for k in self.meta["session"])
        self.geom = {}
        if os.path.exists(geom_path):
            self.geom = {int(k): v for k, v in json.load(open(geom_path)).items()}
        self._bg = {}
        self._dirs = {}

    # ---------- 基础量 ----------
    def angles(self, sid):
        return np.asarray(self.meta["session"][str(sid)]["angles"], float)

    def elevation_deg(self, sid, ray):
        """俯仰角（度）。注意取负号——这是解出来的约定。"""
        return -self.angles(sid)[np.asarray(ray, int)]

    def azimuth_deg(self, azimuth_field):
        """azimuth 字段（0.01° 单位）-> 度。"""
        return np.asarray(azimuth_field, float) / 100.0

    def n_cols(self, sid):
        return IMG_SHAPE[sid][1]

    def col_of(self, sid, azimuth_deg):
        """方位角 -> 背景图列号。"""
        res = 360.0 / self.n_cols(sid)
        return np.clip(np.round(np.asarray(azimuth_deg) / res).astype(int),
                       0, self.n_cols(sid) - 1)

    # ---------- 单帧球坐标 -> 传感器局部 ----------
    def local_from_fields(self, sid, ray, azimuth_field, distance):
        """用 PLY 自带字段重建传感器局部坐标（米）。"""
        el = np.radians(self.elevation_deg(sid, ray))
        az = np.radians(self.azimuth_deg(azimuth_field))
        r = np.asarray(distance, float) * K_M
        return np.stack([r * np.cos(el) * np.cos(az),
                         r * np.cos(el) * np.sin(az),
                         r * np.sin(el)], 1)

    # ---------- 几何参数 ----------
    def fit(self, sid, frames=range(31), max_pts=40000):
        """用若干真实帧拟合该 session 的 R, t（尺度固定为 1，因为已用米制）。"""
        Ls, Ps = [], []
        for f in frames:
            p = os.path.join(LIDAR_DIR, f"{f:06d}.ply")
            if not os.path.exists(p):
                continue
            a, xyz = load_frame(p)
            m = a["id"] == sid
            if m.sum() < 50:
                continue
            Ls.append(self.local_from_fields(sid, a["ray"][m],
                                             a["azimuth"][m], a["distance"][m]))
            Ps.append(xyz[m].astype(np.float64))
        if not Ls:
            raise ValueError(f"session {sid} 在给定帧里没有点")
        L, P = np.concatenate(Ls), np.concatenate(Ps)
        idx = np.random.RandomState(0).choice(len(L), min(len(L), max_pts),
                                              replace=False)
        k, R, t = kabsch(L[idx], P[idx])
        res = P[idx] - (k * (L[idx] @ R.T) + t)
        return {"k": k, "R": R.tolist(), "t": t.tolist(),
                "kabsch_rms": float(np.sqrt((res ** 2).sum(1).mean())),
                "img_shape": list(IMG_SHAPE[sid])}

    def params(self, sid):
        if sid not in self.geom:
            self.geom[sid] = self.fit(sid)
        g = self.geom[sid]
        return g["k"], np.asarray(g["R"], float), np.asarray(g["t"], float)

    # ---------- 光束方向 ----------
    def beam_dirs(self, sid):
        """(n_rows, n_cols, 3) 每个光束在世界坐标下的单位方向。"""
        if sid in self._dirs:
            return self._dirs[sid]
        nr, nc = IMG_SHAPE[sid]
        _, R, _ = self.params(sid)
        el = np.radians(self.elevation_deg(sid, np.arange(nr)))
        az = np.radians(np.arange(nc) * (360.0 / nc))
        E, A = np.meshgrid(el, az, indexing="ij")
        local = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
        d = local.reshape(-1, 3) @ R.T
        self._dirs[sid] = d.reshape(nr, nc, 3)
        return self._dirs[sid]

    # ---------- 背景距离图 ----------
    def background_range(self, sid):
        """(n_rows, n_cols) 背景距离，单位米。0 表示无回波。"""
        if sid not in self._bg:
            img = np.load(os.path.join(BG_DIR, f"{sid}.npy")).astype(np.float32)
            self._bg[sid] = img * K_M
        return self._bg[sid]

    def background_world(self, sid, max_xy=120.0):
        """背景距离图 -> 世界坐标点云（已按 max_xy 裁剪）。"""
        r = self.background_range(sid)
        _, _, t = self.params(sid)
        d = self.beam_dirs(sid)
        ok = r > 0
        P = d[ok] * r[ok][:, None] + t
        if max_xy is not None:
            P = P[(np.abs(P[:, 0] - t[0]) < max_xy) &
                  (np.abs(P[:, 1] - t[1]) < max_xy)]
        return P.astype(np.float32)

    # ---------- 诊断 ----------
    def validate(self, frame=0):
        """用一帧真实点云核对每个 session 的还原精度。"""
        p = os.path.join(LIDAR_DIR, f"{frame:06d}.ply")
        a, xyz = load_frame(p)
        rows = []
        for sid in self.sessions:
            if sid not in IMG_SHAPE:
                continue
            m = a["id"] == sid
            if m.sum() < 200:
                continue
            k, R, t = self.params(sid)
            ray = a["ray"][m].astype(int)
            azd = self.azimuth_deg(a["azimuth"][m])
            c = self.col_of(sid, azd)
            r_img = self.background_range(sid)[ray, c]
            el = np.radians(self.elevation_deg(sid, ray))
            az = np.radians(c * (360.0 / self.n_cols(sid)))
            Li = np.stack([r_img * np.cos(el) * np.cos(az),
                           r_img * np.cos(el) * np.sin(az),
                           r_img * np.sin(el)], 1)
            Pi = k * (Li @ R.T) + t
            d3 = np.linalg.norm(Pi - xyz[m], axis=1)
            rows.append({
                "session": sid, "n": int(m.sum()),
                "kabsch_rms": self.geom.get(sid, {}).get("kabsch_rms"),
                "bg_med3d": float(np.median(d3)),
                "bg_p75": float(np.percentile(d3, 75)),
                "frac_02m": float(np.mean(d3 < 0.2)),
                "note": BROKEN.get(sid, ""),
            })
        return rows


if __name__ == "__main__":
    s = M4Sensors()
    print(f"{'session':>8} {'点数':>8} {'Kabsch':>8} {'背景中位':>9} {'p75':>8} {'<0.2m':>7}  备注")
    for r in s.validate():
        print(f"{r['session']:>8} {r['n']:>8} "
              f"{(r['kabsch_rms'] or 0):8.3f} {r['bg_med3d']:9.3f} "
              f"{r['bg_p75']:8.3f} {100*r['frac_02m']:6.0f}%  {r['note']}")
