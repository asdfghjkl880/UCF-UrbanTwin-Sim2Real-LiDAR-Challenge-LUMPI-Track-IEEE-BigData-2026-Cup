"""把最新的 declaration.txt 注入到已有提交包（保持 predictions/synthetic 不变）。

声明正文由 scripts/predict_pp.py 里的 DECL 生成，改完声明后用它刷新提交包，
避免重跑 GPU 推理。

用法:
  py scripts/refresh_decl.py data/out/submission_v20.zip
"""
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import predict_pp as P  # noqa: E402


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "data/out/submission_v20.zip")
    tmp = src + ".tmp"
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        payload = {n: z.read(n) for n in names if n != "declaration.txt"}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as o:
        for n in names:
            o.writestr(n, P.DECL if n == "declaration.txt" else payload[n])
    os.replace(tmp, src)
    print(f"刷新 declaration.txt ({len(P.DECL)} 字节) -> {src}")


if __name__ == "__main__":
    main()
