#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-PREDICHATE_ENCODERS}"

if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load miniconda 2>/dev/null || true
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not in PATH. Load miniconda first." >&2
  exit 1
fi

if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/environment-encoders.yml"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env remove -n "$ENV_NAME" -y
fi

conda env create -f "$ENV_FILE" -n "$ENV_NAME"
conda run -n "$ENV_NAME" python - <<'PY'
import sys
import pandas
import torch
import transformers

if not torch.__version__.startswith("2.6.0+cu124") or torch.version.cuda != "12.4":
    raise SystemExit(f"Unexpected torch build: {torch.__version__}, CUDA={torch.version.cuda}")
if tuple(map(int, transformers.__version__.split(".")[:2])) < (4, 55):
    raise SystemExit(f"Unexpected transformers version: {transformers.__version__}")
print("Python:", sys.executable)
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("pandas:", pandas.__version__)
PY

echo "Encoder environment '$ENV_NAME' created successfully."
