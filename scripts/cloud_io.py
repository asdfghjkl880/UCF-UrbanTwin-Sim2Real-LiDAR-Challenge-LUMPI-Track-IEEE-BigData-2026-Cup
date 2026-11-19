"""真实输入帧点云的唯一入口 + 打包自检。

为什么单独一个模块（教训）：
打包脚本 `mix_json.py` / `pack_fused.py` 曾经拿随包提交的 `synthetic/*.bin`
（那是给 realism 用的合成帧）去统计"框内点数 / 填充率 / 点质心"，而预测其实
建立在真实测试帧 `detection_test_frames/<fid>.bin` 上。两边文件名、帧数、点云
格式都"看起来兼容"，所以这个错误在本地评测里完全不可见（本地评测走的是正确的
帧），只在线上暴露：真实目标的框内点数为 0，分数被乘到 0.01 保底，
Person AP 从 0.2666 掉到 0.0034。

因此这里把"读哪朵云"收成一个入口，并在打包时打印逐类的点支撑统计；
一旦接错帧，`check_support` 会直接把打包打断。
"""
import os

import numpy as np

__all__ = ["read_cloud", "frame_cloud", "support_diag", "check_support", "report_support"]


def read_cloud(path):
    """读一帧点云，返回 (N,3) float64。通道约定与 predict_pp.read_bin 一致。"""
    a = np.fromfile(path, dtype=np.float32)
    if a.size % 4 == 0 and a.size % 3 != 0:
        a = a.reshape(-1, 4)[:, :3]
    else:
        a = a.reshape(-1, 3)
    return a.astype(np.float64)


def frame_cloud(fid, root=None):
    """读某一帧的**真实输入点云**。找不到就报错，绝不静默回退到合成帧。"""
    import pp_config as C
    root = root or C.TEST_FRAMES_DIR
    p = os.path.join(root, "%s.bin" % fid)
    if not os.path.exists(p):
        raise FileNotFoundError(
            "缺少输入帧点云 %s；重排序必须在预测所在的真实帧上做，"
            "不能用 synthetic/*.bin 顶替" % p)
    return read_cloud(p)


def support_diag(rows, pts, acc, per_class=60, seed=1234):
    """抽样统计每类"框内点数 n / 填充率 fill"，累加进 acc{类: [n, fill, cnt]}。

    rows 为 [{'box':[7], 'score':..., 'class':...}]。抽样固定随机种子，可复现。
    """
    from rerank_pts import box_feats
    from scipy.spatial import cKDTree

    if not rows or pts is None or not len(pts):
        return
    tree = cKDTree(pts)
    rng = np.random.RandomState(seed)
    by_cls = {}
    for i, r in enumerate(rows):
        by_cls.setdefault(r["class"], []).append(i)
    for cls, idx in by_cls.items():
        take = idx if len(idx) <= per_class else list(rng.choice(idx, per_class, replace=False))
        for i in take:
            b = rows[i]["box"]
            fe = box_feats(pts, tree, (b[0], b[1], b[2], b[3], b[4], b[5], b[6]))
            slot = acc.setdefault(cls, [0.0, 0.0, 0])
            slot[0] += float(fe[0])          # n
            slot[1] += float(fe[2])          # fill
            slot[2] += 1


def report_support(acc, title="点支撑自检"):
    """打印逐类平均框内点数与填充率，返回 {类: (平均点数, 平均填充率)}。"""
    if not acc:
        print("[%s] 无样本" % title)
        return {}
    print("[%s] 类          平均框内点数  平均填充率    样本" % title)
    per = {}
    for cls in sorted(acc):
        n, f, c = acc[cls]
        per[cls] = (n / max(c, 1), f / max(c, 1))
        print("[%s] %-11s %10.2f %11.3f %7d" % (title, cls, n / max(c, 1), f / max(c, 1), c))
    return per


def check_support(acc, min_mean_n=1.0):
    """点支撑接近全零 -> 说明点云与预测不同源，直接中断打包。"""
    per = report_support(acc)
    if not per:
        raise RuntimeError("点支撑自检没有样本，无法确认点云来源")
    worst = min(v[0] for v in per.values())
    avg = sum(v[0] for v in per.values()) / len(per)
    if avg < min_mean_n:
        raise RuntimeError(
            "点支撑自检失败：各类平均框内点数 %.2f < %.2f，"
            "点云与预测很可能不同源（检查是否误用了 synthetic/*.bin）" % (avg, min_mean_n))
    print("[点支撑自检] 通过：最小 %.2f 平均 %.2f 点/框" % (worst, avg))
    return per
