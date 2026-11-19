"""用点云实测特征对预测框重排序（不改模型权重）。

诊断结论：Van / Truck / Person 的"排序效率"(AP / 最大召回) 只有 15~38%，
而 Car 高达 93%。AP 只由「TP 排在第几位」决定，尾部假阳性免费。所以这里
不删框，只用点云统计量重新加权分数。

特征（每个预测框，用预测的朝向把场景点投到框坐标）:
  n    框内点数
  m    n * (r/20)^2，补偿点密度 ∝ 1/r^2 的距离归一化点数
  fill n/(L*W*H) 归一化后 = 体积填充率
  zr   实测 z 跨度 / 预测 H
  lr   实测 L / 预测 L      (遮挡越完整越接近 1)
  wr   实测 W / 预测 W

用法:
  py scripts/rerank_pts.py --npz data/out/_sbg3_final.npz.npy
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "lumpi_starting_kit", "scoring_program"))

import pp_config as C  # noqa: E402
import fastap2 as F  # noqa: E402
from ap1 import ap40  # noqa: E402
from detection_eval import _iou_3d  # noqa: E402

NAMES = C.CLASS_NAMES
FI = ["n", "m", "fill", "zr", "lr", "wr", "r", "mL", "mW", "mH", "pH",
      "ya", "el", "tp", "zc", "mx", "my", "mz", "dy"]


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z if hasattr(z, "keys") else z.item()


def box_feats(pts, tree, box):
    x, y, zz, L, W, H, yaw = box
    r = float(np.hypot(x, y))
    rad = 0.5 * float(np.sqrt(L * L + W * W + H * H)) + 0.3
    idx = tree.query_ball_point([x, y, zz], rad)
    if len(idx) < 3:
        return [len(idx), len(idx) * (r / 20.0) ** 2, 0.0, 0.0, 0.0, 0.0, r,
                0.0, 0.0, 0.0, H, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    q = pts[idx] - np.array([x, y, zz])
    c, s = np.cos(yaw), np.sin(yaw)
    lx = c * q[:, 0] + s * q[:, 1]
    ly = -s * q[:, 0] + c * q[:, 1]
    m = (np.abs(lx) <= L / 2) & (np.abs(ly) <= W / 2) & (np.abs(q[:, 2]) <= H / 2)
    n = int(m.sum())
    if n < 3:
        return [n, n * (r / 20.0) ** 2, 0.0, 0.0, 0.0, 0.0, r, 0.0, 0.0, 0.0, H,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ax, ay, az = lx[m], ly[m], q[m, 2]
    ml = float(np.percentile(ax, 98) - np.percentile(ax, 2))
    mw = float(np.percentile(ay, 98) - np.percentile(ay, 2))
    mh = float(np.percentile(az, 98) - np.percentile(az, 2))
    vol = max(L * W * H, 1e-3)
    # 朝向一致性：点云 xy 主轴 vs 预测 yaw（车辆有 180 度歧义，用 cos(2Δ)）
    xy = np.stack([ax, ay], 1)
    xy = xy - xy.mean(0)
    cov = xy.T @ xy / max(n - 1, 1)
    ev, evec = np.linalg.eigh(cov)
    el = float(np.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9)))     # 细长比 0..1
    ang = float(np.arctan2(evec[1, 1], evec[0, 1]))
    ya = abs(float(np.cos(2.0 * (ang - yaw))))
    tp = float(((az > 0.5 * H * 0.55).sum()) / max(n, 1))          # 顶面点占比
    zc = 1.0 - min(abs(float(az.mean())) / max(H, 1e-3), 1.0)      # 点云质心居中度
    # 点云质心在框坐标系下的偏移（用于中心修正；世界坐标 = 中心 + R^T(mx,my)）
    dy = (ang - yaw + 0.5 * np.pi) % np.pi - 0.5 * np.pi   # 主轴与预测 yaw 的差（±90°）
    return [n, n * (r / 20.0) ** 2, min(n / vol * 6.0, 3.0),
            mh / max(H, 1e-3), ml / max(L, 1e-3), mw / max(W, 1e-3), r,
            ml, mw, mh, H, ya, el, tp, zc,
            float(ax.mean()), float(ay.mean()), float(az.mean()), float(dy)]


def build_cache(z, clouds, cache_path=""):
    """每个 (frame, class) 预计算 IoU 候选对，之后换权重只重跑贪心匹配。
    支持断点续建：每 10 帧落盘一次。"""
    cache = {}
    if cache_path and os.path.exists(cache_path + ".part"):
        cache = np.load(cache_path + ".part", allow_pickle=True).item()
        print(f"续建 {cache_path}.part  ({len({k[0] for k in cache})} 帧)", flush=True)
    done = {k[0] for k in cache}
    fids = [f for f in sorted(z) if f not in done]
    for n_done, fid in enumerate(fids, 1):
        b, s, c, g = z[fid]
        b = np.asarray(b, np.float64)
        s = np.asarray(s, np.float64).ravel()
        c = np.asarray(c).astype(np.int64).ravel()
        p = os.path.join(clouds, fid + ".bin")
        if os.path.exists(p):
            pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        else:
            pts = np.zeros((0, 3))
        tree = cKDTree(pts) if len(pts) else None
        feat = np.zeros((len(b), len(FI)))
        if tree is not None:
            for i in range(len(b)):
                feat[i] = box_feats(pts, tree, b[i])
        gb = np.asarray(g, np.float64).reshape(-1, 8)
        for ci, nm in enumerate(NAMES):
            pi = np.flatnonzero(c == ci)
            gi = np.flatnonzero(gb[:, 0] == ci) if len(gb) else np.zeros(0, int)
            gg = gb[gi, 1:8] if len(gi) else np.zeros((0, 7))
            a = bb = vv = np.zeros(0)
            if len(pi) and len(gi):
                d = np.hypot(b[pi][:, None, 0] - gg[None, :, 0],
                             b[pi][:, None, 1] - gg[None, :, 1])
                half = (0.5 * np.hypot(gg[:, 3], gg[:, 4])[None, :]
                        + 0.5 * np.hypot(b[pi][:, 3], b[pi][:, 4])[:, None])
                mm = d < half * 1.05
                a, bb = np.nonzero(mm)
                if len(a):
                    vv = np.array([_iou_3d(b[pi][a[k]], gg[bb[k]])
                                   for k in range(len(a))])
                    keep = vv > 0.02
                    a, bb, vv = a[keep], bb[keep], vv[keep]
            cache[(fid, nm)] = dict(sc=s[pi], feat=feat[pi], ngt=len(gi),
                                    iou=(a, bb, vv), npred=len(pi),
                                    pb=b[pi].astype(np.float32),
                                    gb=gg.astype(np.float32))
        print(f"  [{n_done}/{len(fids)}] {fid} ok", flush=True)
        if cache_path and n_done % 10 == 0:
            np.save(cache_path + ".part", cache, allow_pickle=True)
    return cache


def greedy(entry, w, thr=0.5):
    a, b, v = entry["iou"]
    tps = np.zeros(len(entry["sc"]), bool)
    if not len(a):
        return tps
    adj = {}
    for aa, bb, vv in zip(a, b, v):
        if vv >= thr:
            adj.setdefault(int(aa), []).append((int(bb), float(vv)))
    used = np.zeros(entry["ngt"], bool)
    for p in np.argsort(-w, kind="stable"):
        lst = adj.get(int(p))
        if not lst:
            continue
        best = None
        for gg, vv in lst:
            if not used[gg] and (best is None or vv > best[1]):
                best = (gg, vv)
        if best is not None:
            used[best[0]] = True
            tps[p] = True
    return tps


def _term(t, param):
    """单项权重因子。param 形如 'n10b1' / 'd20b1' / 'fill10' / 'zr0.5' /
    'H2.0x0.22'（实测高度高斯）/ 'L4.6x0.9'（实测长度高斯）/ 'mL8'（长度比幂）。"""
    n, m = t[:, 0], t[:, 1]
    if param[0] == "n":
        k, b = param[1:].split("b")
        return (n / (n + float(k))) ** float(b)
    if param[0] == "d":
        k, b = param[1:].split("b")
        return (m / (m + float(k))) ** float(b)
    if param.startswith("H"):
        mu, sd = param[1:].split("x")
        return np.exp(-0.5 * ((t[:, 9] - float(mu)) / float(sd)) ** 2)
    if param.startswith("L"):
        mu, sd = param[1:].split("x")
        return np.exp(-0.5 * ((t[:, 7] - float(mu)) / float(sd)) ** 2)
    if param.startswith("W"):
        mu, sd = param[1:].split("x")
        return np.exp(-0.5 * ((t[:, 8] - float(mu)) / float(sd)) ** 2)
    if param.startswith("zr"):
        return np.clip(t[:, 3], 0.05, 1.0) ** float(param[2:] or 1.0)
    if param.startswith("lr"):
        return np.clip(t[:, 4], 0.05, 1.0) ** float(param[2:] or 1.0)
    if param.startswith("wr"):
        return np.clip(t[:, 5], 0.05, 1.0) ** float(param[2:] or 1.0)
    if param.startswith("fill"):
        return np.clip(t[:, 2], 0.05, 3.0) ** float(param[4:] or 1.0)
    if param.startswith("ya"):     # 朝向一致性
        return np.clip(t[:, 11], 0.02, 1.0) ** float(param[2:] or 1.0)
    if param.startswith("el"):     # 细长比
        return np.clip(t[:, 12], 0.02, 1.0) ** float(param[2:] or 1.0)
    if param.startswith("tp"):     # 顶面点占比
        return np.clip(t[:, 13], 0.02, 1.0) ** float(param[2:] or 1.0)
    if param.startswith("zc"):     # 点云质心居中度
        return np.clip(t[:, 14], 0.02, 1.0) ** float(param[2:] or 1.0)
    raise ValueError(param)


def wfun(name, fe):
    """name 用 '*' 连接多个因子，例如 'n20b1*H2.0x0.22'。"""
    if name == "base":
        return np.ones(len(fe))
    w = np.ones(len(fe))
    for param in name.split("*"):
        w = w * _term(fe, param)
    return np.maximum(w, 0.01)


CAND = (["base"] +
        [f"n{k}b{b}" for k in (2, 5, 10, 20, 50, 100, 200) for b in (0.5, 1.0, 1.5)] +
        [f"d{k}b{b}" for k in (5, 10, 20, 50, 100, 200) for b in (0.5, 1.0, 1.5)] +
        ["zr1", "zr2", "lr1", "lr2", "wr1", "wr2", "fill1", "fill0.5",
         "ya1", "ya2", "el1", "tp1", "tp2", "zc1", "zc2"] +
        [f"n20b1*{t}" for t in ("zr1", "lr1", "wr1", "fill1")])

# 实测尺寸高斯项（按类给形状先验；Van/Truck/Person 的判别主要靠实测高度/长度）
DIM_CAND = {
    "Van":    ["H2.03x0.18", "H2.03x0.25", "L4.6x1.0", "H2.03x0.18*L4.6x1.0"],
    "Car":    ["H1.66x0.15", "L4.4x0.8", "H1.66x0.15*L4.4x0.8"],
    "Truck":  ["H3.1x0.45", "L8.2x2.0", "H3.1x0.45*L8.2x2.0"],
    "Bus":    ["H2.75x0.5", "L7.8x2.0"],
    "Person": ["H1.85x0.15", "L1.2x0.5", "H1.85x0.15*L1.2x0.5"],
    "Bicycle": ["H1.9x0.15", "L1.95x0.15"],
    "Motorcycle": ["H1.85x0.15", "L1.95x0.15"],
}


def ev(cache, nm, keys, name):
    S, T, ngt = [], [], 0
    for k in keys:
        e = cache[(k, nm)]
        ngt += e["ngt"]
        if not e["npred"]:
            continue
        w = e["sc"] * wfun(name, e["feat"])
        S.append(w)
        T.append(greedy(e, w))
    if not S or ngt == 0:
        return 0.0
    return ap40(np.concatenate(S), np.concatenate(T), ngt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "data/out/_sbg3_final.npz.npy"))
    ap.add_argument("--clouds", default=os.path.join(ROOT, "data/eval_m4_holdout"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "data/out/_rr_pts.npy"))
    ap.add_argument("--cfg", default=os.path.join(ROOT, "reports/rerank_pts.json"))
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.cache):
        cache = np.load(a.cache, allow_pickle=True).item()
        print(f"载入缓存 {a.cache}  ({len(cache)} 项)")
    else:
        z = load_npz(a.npz)
        cache = build_cache(z, a.clouds, a.cache)
        np.save(a.cache, cache, allow_pickle=True)
        print(f"-> {a.cache}")
        if os.path.exists(a.cache + ".part"):
            os.remove(a.cache + ".part")
    if a.build_only:
        return

    fids = sorted({k[0] for k in cache})
    A = set(fids[::2])
    cfg, cfg_up = {}, {}
    print(f'{"class":11s} {"n_gt":>5} {"A:base":>7} {"B:base":>7} '
          f'{"A:best":>7} {"B:best":>7}  param')
    for nm in NAMES:
        keys = [k for k in fids if (k, nm) in cache]
        if not keys or sum(cache[(k, nm)]["ngt"] for k in keys) == 0:
            continue
        ngt = sum(cache[(k, nm)]["ngt"] for k in keys)
        ka = [k for k in keys if k in A]
        kb = [k for k in keys if k not in A]
        ab, bb = ev(cache, nm, ka, "base"), ev(cache, nm, kb, "base")
        cands = CAND[1:] + DIM_CAND.get(nm, []) + [
            f"n20b1*{t}" for t in DIM_CAND.get(nm, [])]
        best = None
        for wn in cands:
            va, vb = ev(cache, nm, ka, wn), ev(cache, nm, kb, wn)
            if va > ab and vb > bb:
                sc = 0.5 * (va + vb)
                if best is None or sc > best[0]:
                    best = (sc, wn, va, vb)
        if best is None:
            print(f'{nm:11s} {ngt:5d} {ab:7.4f} {bb:7.4f} {"-":>7} {"-":>7}  保持默认')
            continue
        _, wn, va, vb = best
        cfg[nm] = wn
        print(f'{nm:11s} {ngt:5d} {ab:7.4f} {bb:7.4f} {va:7.4f} {vb:7.4f}  {wn}')
        cfg_up[nm] = [wn, round(va - ab, 4), round(vb - bb, 4)]

    m0 = m1 = 0.0
    nc = 0
    for nm in NAMES:
        keys = [k for k in fids if (k, nm) in cache]
        if not keys or sum(cache[(k, nm)]["ngt"] for k in keys) == 0:
            continue
        nc += 1
        m0 += ev(cache, nm, keys, "base")
        m1 += ev(cache, nm, keys, cfg.get(nm, "base"))
    m0 /= max(nc, 1)
    m1 /= max(nc, 1)
    print(f"\n全量 mAP  基线 {m0:.4f}  重排序 {m1:.4f}  ({m1-m0:+.4f}, {nc} 类)")
    json.dump(cfg_up, open(a.cfg, "w"), indent=1)
    print(f"-> {a.cfg}")


if __name__ == "__main__":
    main()
