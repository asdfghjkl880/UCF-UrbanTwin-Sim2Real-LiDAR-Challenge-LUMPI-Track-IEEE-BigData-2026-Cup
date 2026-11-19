#!/usr/bin/env bash
# =============================================================================
# UrbanTwin Sim2Real LiDAR Challenge — LUMPI Track
# 数据下载脚本（Linux / WSL2 / Git Bash 均可，需要 curl + 7z）
#
# 用法：
#   bash fetch_data.sh labels          # 只下标签 1.04GB（先跑这个，最划算）
#   bash fetch_data.sh starter         # 小文件全家桶 ~3.2GB（labels+sample+map+参考点云）
#   bash fetch_data.sh refcloud        # 路口静态地图点云 1.15GB（合成背景底座）
#   bash fetch_data.sh sample          # 下 LUMPI test_data.zip 838MB（摸清格式）
#   bash fetch_data.sh map             # 下 HD map 1.6MB + meta.json
#   bash fetch_data.sh lidar 5         # 下 Measurement5 的 LiDAR 全部分卷（9 卷，36.5GB）
#   bash fetch_data.sh utlumpi         # 下 UT-LUMPI 25 个分卷（47.5GB）
#   bash fetch_data.sh extract <dir>   # 解开某个目录下所有分卷压缩包
#
# 依赖：curl；7-Zip（Linux: sudo apt install p7zip-full / Windows: winget install 7zip.7zip）
# 断点续传：所有下载都用 curl -C -，中断后重跑同一条命令即可。
# =============================================================================
set -euo pipefail

ROOT="${DATA_ROOT:-/d/lumpi_data}"          # 数据根目录，可用 DATA_ROOT 覆盖
VAULT="https://data.uni-hannover.de/vault/ikg/busch/LUMPI"
DV_API="https://dataverse.harvard.edu/api"
UT_DOI="doi:10.7910/DVN/D9SSWD"

mkdir -p "$ROOT"

dl() {  # dl <url> <outfile>
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  echo ">>> $out"
  curl -fL -C - --retry 5 --retry-delay 5 --connect-timeout 30 \
       --progress-bar -o "$out" "$url"
}

case "${1:-}" in

  labels)
    dl "$VAULT/labels.zip" "$ROOT/LUMPI/labels.zip"
    ;;

  sample)
    dl "$VAULT/test_data.zip" "$ROOT/LUMPI/test_data.zip"
    ;;

  map)
    dl "$VAULT/HD_map/lumpi_lines_arcs.xodr"  "$ROOT/LUMPI/HD_map/lumpi_lines_arcs.xodr"
    dl "$VAULT/HD_map/lumpi_polylines.csv"    "$ROOT/LUMPI/HD_map/lumpi_polylines.csv"
    dl "$VAULT/meta.json"                     "$ROOT/LUMPI/meta.json"
    ;;

  refcloud)
    # 路口静态地图点云（UTM）——合成场景的静态背景底座，性价比最高
    dl "$VAULT/Reference_map_point_cloud/LUMPI_reference_cloud.zip" \
       "$ROOT/LUMPI/Reference_map_point_cloud/LUMPI_reference_cloud.zip"
    ;;

  starter)
    # 一批小文件打包：标签 + 样本 + 地图 + 参考云（约 3.2GB，10 分钟内下完）
    "$0" labels
    "$0" sample
    "$0" map
    "$0" refcloud
    ;;

  lidar)
    M="${2:?用法: bash fetch_data.sh lidar <0-6>}"
    # 每个 Measurement 的 lidar 分卷数量不同（M6 有 14 卷），先探测存在性
    n=1
    for i in $(seq 1 20); do
      if curl -sfI "$VAULT/Measurement${M}/lidar.zip.$(printf '%03d' $i)" >/dev/null; then
        n=$i
      else
        break
      fi
    done
    echo ">>> Measurement${M} 共 ${n} 个 LiDAR 分卷"
    dl "$VAULT/Measurement${M}/background.zip" \
       "$ROOT/LUMPI/Measurement${M}/background.zip"
    for i in $(seq 1 "$n"); do
      part=$(printf '%03d' "$i")
      dl "$VAULT/Measurement${M}/lidar.zip.${part}" \
         "$ROOT/LUMPI/Measurement${M}/lidar.zip.${part}"
    done
    ;;

  utlumpi)
    # 通过 Dataverse API 拿到 25 个分卷的真实 file id，逐个续传下载
    python3 - "$DV_API" "$UT_DOI" "$ROOT/UT-LUMPI" <<'PY'
import json, os, subprocess, sys, urllib.request

api, doi, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
meta = json.load(urllib.request.urlopen(
    f"{api}/datasets/:persistentId/?persistentId={doi}", timeout=60))
files = meta["data"]["latestVersion"]["files"]
print(f"UT-LUMPI: {len(files)} 个文件")
for f in files:
    df = f["dataFile"]
    fid, name, size = df["id"], df["filename"], df.get("filesize", 0)
    dst = os.path.join(outdir, name)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print(f"  跳过 {name}")
        continue
    print(f"  下载 {name} ({size/2**20:.0f} MB)")
    subprocess.run(["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "5",
                    "--progress-bar", "-o", dst,
                    f"{api}/access/datafile/{fid}"], check=True)
PY
    ;;

  extract)
    D="${2:?用法: bash fetch_data.sh extract <dir>}"
    cd "$D"
    for f in *.zip.001 *.zip; do
      [ -e "$f" ] || continue
      echo ">>> 解压 $f"
      7z x -y "$f" || 7zz x -y "$f"
    done
    ;;

  *)
    sed -n '2,20p' "$0"
    ;;
esac
