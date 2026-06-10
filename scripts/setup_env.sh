#!/usr/bin/env bash
# 注册 Jupyter Kernel，供 Cursor / Jupyter 选择
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$ROOT/.conda/bin:$PATH"

if [[ ! -x "$ROOT/.conda/bin/python" ]]; then
  echo "错误: 未找到 $ROOT/.conda/bin/python"
  echo "请先安装 Miniconda 到项目 .conda 目录。"
  exit 1
fi

"$ROOT/.conda/bin/python" -m ipykernel install --user --name used-car-pricing --display-name "Python (used-car-pricing)"
"$ROOT/.conda/bin/python" -c "import xgboost; print('xgboost OK', xgboost.__version__)"
echo ""
echo "完成。在 Notebook 里选择 Kernel: Python (used-car-pricing)"
