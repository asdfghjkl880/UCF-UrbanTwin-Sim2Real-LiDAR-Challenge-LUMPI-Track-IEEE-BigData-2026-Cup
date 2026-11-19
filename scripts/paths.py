#!/usr/bin/env python3
"""
项目路径中心配置。

2026-11-09 目录重整后的结构：

    data/
      LUMPI/                        <- 官方原始数据集全部收在这里
        HD_map/                     lumpi_polylines.csv + 2 个 xodr
        labels/Label/               7 个 measurement 的 Label.csv + Map/ + meta.json
        test_data/test_data/        7 x 31 帧真实点云
        Reference_map_point_cloud/
          LUMPI_reference_cloud/    全量静态地图（6,530 万点）
          LUMPI_preview_reference_cloud/   1/16 预览版（408 万点）
        Measurement4/               官方 M4 原始 lidar 分卷（下载中）
        meta.json                   传感器参数（7 x 5 个 LiDAR 的线数/角度表）
      derived/                      <- 本项目派生产物，**不进官方 LUMPI 目录**
        m4_background/              bg_m4_*.npy / bgsrc_m4_*.npy 静态背景距离图
        m4_geometry.json            自己拟合的 M4 传感器几何
        lumpi_forbidden_frames_mapping.csv  赛事帧号 -> 原始帧号映射
      train/                        合成+真实训练集、checkpoint
      synthetic/                    提交用的 50 帧合成点云
      reference_data_local/         本地评分基准（31 帧 + GT）
      out/                          提交包

以后调整目录结构，只改这一个文件。

    from paths import LABEL_DIR, TEST_DATA, REF_CLOUD
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LUMPI = os.path.join(DATA, "LUMPI")

# ---- 官方原始数据集 ----
LABEL_DIR = os.path.join(LUMPI, "labels", "Label")        # 7 个 Label.csv + Map/
META_JSON = os.path.join(LUMPI, "meta.json")              # 传感器参数
TEST_DATA = os.path.join(LUMPI, "test_data", "test_data")  # 7 x 31 帧真实点云
HD_MAP = os.path.join(LUMPI, "HD_map")                    # xodr + polylines
MAP_ROOT = os.path.join(LUMPI, "Reference_map_point_cloud")
REF_CLOUD = os.path.join(
    MAP_ROOT, "LUMPI_reference_cloud", "LUMPI_reference_cloud.ply")
PREVIEW_CLOUD = os.path.join(
    MAP_ROOT, "LUMPI_preview_reference_cloud",
    "LUMPI_preview_reference_cloud.ply")
M4_RAW = os.path.join(LUMPI, "Measurement4")              # 原始 lidar 分卷

# ---- 赛事起跑包 ----
KIT = os.path.join(ROOT, "lumpi_starting_kit")
TEST_FRAMES = os.path.join(KIT, "detection_test_frames")
FORBIDDEN = os.path.join(KIT, "forbidden_frames.txt")

# ---- 本项目派生产物（与官方原始数据严格分开，不写进 LUMPI 目录）----
DERIVED = os.path.join(DATA, "derived")
M4_BG_DIR = os.path.join(DERIVED, "m4_background")        # 静态背景距离图
M4_GEOM = os.path.join(DERIVED, "m4_geometry.json")       # 拟合的传感器几何
FORBIDDEN_MAP = os.path.join(
    DERIVED, "lumpi_forbidden_frames_mapping.csv")        # 赛事帧号映射

# ---- 本项目产物 ----
LOCAL_BENCH = os.path.join(DATA, "reference_data_local")
TRAIN_DIR = os.path.join(DATA, "train")
CKPT_DIR = os.path.join(TRAIN_DIR, "ckpt")
SYNTH = os.path.join(DATA, "synthetic")
OUT_DIR = os.path.join(DATA, "out")
REPORTS = os.path.join(ROOT, "reports")
LOGS = os.path.join(REPORTS, "logs")


def label_csv(measurement):
    """返回某个 measurement 的 Label.csv 路径，兼容官方拼写错误 Measurment0。"""
    p = os.path.join(LABEL_DIR, measurement, "Label.csv")
    if os.path.exists(p):
        return p
    alt = measurement.replace("Measurement", "Measurment")
    return os.path.join(LABEL_DIR, alt, "Label.csv")


MEASUREMENTS = ["Measurment0", "Measurement1", "Measurement2", "Measurement3",
                "Measurement4", "Measurement5", "Measurement6"]


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.isupper() and isinstance(v, str):
            print(f"{k:16s} {'OK  ' if os.path.exists(v) else '缺失'} {v}")
