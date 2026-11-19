@echo off
REM ============================================================
REM  环境快速激活（Windows CMD / PowerShell 双击可用）
REM  用法：在项目根目录执行  scripts\env.bat
REM  之后当前窗口内 python / pip / conda 都指向 lidar310 环境
REM ============================================================

set "ANACONDA_HOME=D:\app\anaconda"
set "CONDA_ENV_NAME=lidar310"
set "CONDA_ENV=D:\app\anaconda\envs\lidar310"
set "PIP_CACHE_DIR=D:\app\anaconda\cache\pip"
set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
set "CONDA_PKGS_DIRS=D:\app\anaconda\pkgs"
set "TORCH_HOME=D:\app\anaconda\cache\torch"
set "HF_HOME=D:\app\anaconda\cache\hf"
REM 临时目录放 D 盘，避免 pip 装大包时写满 C:\...\Temp
set "TMPDIR=D:\app\anaconda\cache\tmp"
set "TEMP=D:\app\anaconda\cache\tmp"
set "TMP=D:\app\anaconda\cache\tmp"

set "PATH=%CONDA_ENV%;%CONDA_ENV%\Scripts;%CONDA_ENV%\Library\bin;%ANACONDA_HOME%;%ANACONDA_HOME%\Scripts;%PATH%"

echo 环境就绪
echo   ANACONDA_HOME = %ANACONDA_HOME%
echo   CONDA_ENV     = %CONDA_ENV%  (%CONDA_ENV_NAME%)
echo   Python        = %CONDA_ENV%\python.exe
"%CONDA_ENV%\python.exe" -c "import sys;print('  版本',sys.version.split()[0])"
"%CONDA_ENV%\python.exe" -c "import numpy,scipy;print('  numpy',numpy.__version__,'| scipy',scipy.__version__)" 2>nul || echo   (numpy/scipy 未安装)
"%CONDA_ENV%\python.exe" -c "import torch;print('  torch',torch.__version__,'| CUDA',torch.cuda.is_available())" 2>nul || echo   (torch 未安装)
echo.
echo 常用命令：
echo   评分  python lumpi_starting_kit\local_eval.py dev_submission.zip reference_data_local
echo   装包  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名
echo   切环境  conda activate lidar310
echo.
cmd /k
