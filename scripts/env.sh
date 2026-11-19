#!/usr/bin/env bash
# 环境快速激活（bash / Git Bash 都能 source）
#
#   source scripts/env.sh
#
# 之后可直接用：py / pipi / conda / evall / train
# 所有路径都是实测存在的，不用再猜。

export ANACONDA_HOME="/d/app/anaconda"
export CONDA_ENV_NAME="lidar310"
export CONDA_ENV="/d/app/anaconda/envs/lidar310"
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 所有缓存重定向到 D 盘，避免占用 C 盘
export PIP_CACHE_DIR="/d/app/anaconda/cache/pip"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export CONDA_PKGS_DIRS="/d/app/anaconda/pkgs"
export TORCH_HOME="/d/app/anaconda/cache/torch"
export HF_HOME="/d/app/anaconda/cache/hf"
# 临时目录也放 D 盘：pip 装大包（torch 2.4GB）时默认写 C:\...\Temp
export TMPDIR="/d/app/anaconda/cache/tmp"
export TEMP='D:\app\anaconda\cache\tmp'
export TMP='D:\app\anaconda\cache\tmp'
mkdir -p "$TMPDIR"

export PY="$CONDA_ENV/python.exe"
[ -x "$PY" ] || export PY="python"

export PATH="$CONDA_ENV:$CONDA_ENV/Scripts:$CONDA_ENV/Library/bin:$ANACONDA_HOME/Scripts:$PATH"

py()    { "$PY" "$@"; }
pipi()  { "$PY" -m pip install -i "$PIP_INDEX_URL" "$@"; }
conda() { "$ANACONDA_HOME/Scripts/conda.exe" "$@"; }

# 跑本地评分：evall <提交zip> [参考目录]
evall() {
  local zip="${1:?用法: evall dev_submission.zip [参考目录]}"
  local ref="${2:-$PROJECT_ROOT/data/reference_data_local}"
  "$PY" "$PROJECT_ROOT/lumpi_starting_kit/local_eval.py" "$zip" "$ref"
}

# 训练：train <脚本.py> [参数...]
train() {
  local s="${1:?用法: train scripts/train_xxx.py [参数...]}"; shift
  "$PY" "$s" "$@"
}

echo "环境就绪"
echo "  ANACONDA_HOME = $ANACONDA_HOME"
echo "  CONDA_ENV     = $CONDA_ENV  ($CONDA_ENV_NAME)"
echo "  PY            = $PY"
"$PY" -c "import sys;print('  Python', sys.version.split()[0])" 2>/dev/null
"$PY" -c "import numpy,scipy;print('  numpy',numpy.__version__,'| scipy',scipy.__version__)" 2>/dev/null || \
  echo "  (numpy/scipy 未安装)"
"$PY" -c "import torch;print('  torch',torch.__version__,'| CUDA',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>/dev/null || \
  echo "  (torch 未安装)"
