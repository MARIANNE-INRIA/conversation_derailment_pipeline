#!/usr/bin/env bash
set -euo pipefail

CHECK_GPU=0
CHECK_MODELS=0
CHECK_ENCODER_DATA=0

for argument in "$@"; do
  case "$argument" in
    --gpu) CHECK_GPU=1 ;;
    --models) CHECK_MODELS=1 ;;
    --encoder-data) CHECK_ENCODER_DATA=1 ;;
    *)
      echo "Usage: $0 [--gpu] [--models] [--encoder-data]" >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "PREDICHATE" ]]; then
  echo "Wrong Conda environment: ${CONDA_DEFAULT_ENV:-none}" >&2
  echo "Activate it first with: conda activate PREDICHATE" >&2
  exit 1
fi

if [[ "$(python -c 'import sys; print(sys.prefix)')" != "${CONDA_PREFIX:-}" ]]; then
  echo "Python is not the interpreter from PREDICHATE: $(command -v python)" >&2
  echo "Reactivate the environment with: conda activate PREDICHATE" >&2
  exit 1
fi

for required_file in scripts/train_llm_forecasting.py scripts/train_deberta.py scripts/train_roberta.py \
  data/train_samplesLLMs.csv data/val_samplesLLMs.csv data/test_samplesLLMs.csv; do
  [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 1; }
done

if (( CHECK_ENCODER_DATA )); then
  for encoder in deberta roberta; do
    for split in train val test; do
      [[ -f "$ROOT_DIR/encoder_data/$encoder/${split}_samples.pt" ]] || {
        echo "Missing encoder data: $ROOT_DIR/encoder_data/$encoder/${split}_samples.pt" >&2
        exit 1
      }
    done
  done
fi

python - <<'PY'
import importlib
import sys

required = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "sentencepiece",
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "huggingface_hub",
]
for name in required:
    importlib.import_module(name)

import peft
import torch
import transformers

print("Python:", sys.executable)
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)

if not torch.__version__.startswith("2.5.1+cu121"):
  raise SystemExit("Expected torch 2.5.1+cu121")
if not transformers.__version__.startswith("4.55."):
    raise SystemExit("Expected transformers 4.55.x")
if not peft.__version__.startswith("0.20."):
    raise SystemExit("Expected peft 0.20.x")

if not torch.cuda.is_available():
    print("GPU: unavailable (use --gpu inside a Slurm GPU allocation)")
else:
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))

PY

if (( CHECK_GPU )); then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("A CUDA GPU is required for --gpu")
if torch.cuda.device_count() < 1:
    raise SystemExit("No CUDA device is visible")
print("CUDA test:", torch.cuda.get_device_name(0), "OK")
PY
fi

if (( CHECK_MODELS )); then
  python - <<'PY'
from huggingface_hub import get_token
from transformers import AutoTokenizer

if not get_token():
    raise SystemExit("No Hugging Face token found. Run: huggingface-cli login")

models = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma3": "google/gemma-3-4b-it",
    "deberta": "microsoft/deberta-v3-base",
    "roberta": "roberta-base",
}
for name, model_id in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Model access: {name} ({model_id}) OK; vocab={tokenizer.vocab_size}")
PY
fi

python -m py_compile scripts/train_llm_forecasting.py scripts/train_deberta.py \
  scripts/train_roberta.py scripts/prepare_encoder_pt.py

echo "Environment checks passed."
