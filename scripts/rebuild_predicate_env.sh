#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-PREDICHATE}"
CONDA_PREFIX_PATH="${2:-}"

if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load miniconda 2>/dev/null || true
  fi
fi

if [ -n "$CONDA_PREFIX_PATH" ]; then
  export PATH="$CONDA_PREFIX_PATH/bin:$PATH"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not in PATH. Load miniconda first or pass its prefix as the second argument."
  exit 1
fi

# Initialize conda in this shell without requiring 'conda init' to have been run.
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi

hash -r

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' exists. Rebuilding it..."
  conda activate "$ENV_NAME"
else
  echo "Creating environment '$ENV_NAME'..."
  conda create -n "$ENV_NAME" python=3.10 -y
  conda activate "$ENV_NAME"
fi

if [[ "$(python -c 'import sys; print(sys.prefix)')" != "$CONDA_PREFIX" ]]; then
  echo "Python is not coming from the active Conda environment."
  echo "  CONDA_PREFIX=$CONDA_PREFIX"
  echo "  python=$(command -v python)"
  echo "  sys.prefix=$(python -c 'import sys; print(sys.prefix)')"
  exit 1
fi

conda run -n "$ENV_NAME" python -m pip uninstall -y \
  torch torchvision torchaudio \
  transformers peft \
  huggingface_hub tokenizers || true

conda remove -y -n "$ENV_NAME" pytorch libtorch pytorch-cuda pytorch-mutex cuda-cudart cuda-runtime || true

conda remove -y -n "$ENV_NAME" torchvision torchaudio || true

rm -rf "$CONDA_PREFIX"/lib/python3.10/site-packages/torch* 2>/dev/null || true
rm -rf "$CONDA_PREFIX"/lib/python3.10/site-packages/torchvision* 2>/dev/null || true
rm -rf "$CONDA_PREFIX"/lib/python3.10/site-packages/torchaudio* 2>/dev/null || true
rm -rf "$CONDA_PREFIX"/lib/python3.10/site-packages/transformers* 2>/dev/null || true
rm -rf "$CONDA_PREFIX"/lib/python3.10/site-packages/peft* 2>/dev/null || true
rm -rf "$CONDA_PREFIX"/lib/python3.10/site-packages/huggingface_hub* 2>/dev/null || true
rm -rf "$CONDA_PREFIX"/lib/python3.10/site-packages/tokenizers* 2>/dev/null || true

# Install the CUDA wheel explicitly. Conda may otherwise solve to a CPU-only
# PyTorch build even when pytorch-cuda is present in the environment.
conda run -n "$ENV_NAME" python -m pip install \
  --no-cache-dir --force-reinstall \
  torch==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

hash -r

conda run -n "$ENV_NAME" python - <<'PY'
import torch

expected = "2.5.1"
if not torch.__version__.startswith(expected):
  raise SystemExit(
    f"Unexpected PyTorch version: {torch.__version__}; expected {expected}. "
    "The environment was not rebuilt correctly."
  )
print("PyTorch version check:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
PY

conda run -n "$ENV_NAME" python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "huggingface_hub>=0.34.0,<1.0" \
  "tokenizers>=0.21,<0.22" \
  transformers==4.55.0 \
  accelerate \
  peft==0.20.0 \
  protobuf==5.29.5

conda run -n "$ENV_NAME" python -m pip install --no-cache-dir --force-reinstall \
  sentencepiece \
  pandas numpy scikit-learn

conda run -n "$ENV_NAME" python - <<'PY'
import torch

if not torch.__version__.startswith("2.5.1+cu121") or torch.version.cuda != "12.1":
    raise SystemExit(
    f"ERROR: wrong PyTorch build remains: {torch.__version__}, "
    f"CUDA={torch.version.cuda}. Expected torch 2.5.1+cu121."
    )
print("PyTorch package check:", torch.__version__)
PY

echo

echo "Environment rebuild complete."
echo "Verifying imports..."
conda run -n "$ENV_NAME" python - <<'PY'
import peft
import google.protobuf
import torch
import transformers
from transformers import get_linear_schedule_with_warmup

if not transformers.__version__.startswith("4.55."):
  raise SystemExit(f"Unexpected transformers version: {transformers.__version__}")
if not peft.__version__.startswith("0.20."):
  raise SystemExit(f"Unexpected peft version: {peft.__version__}")

print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('scheduler OK')
PY
