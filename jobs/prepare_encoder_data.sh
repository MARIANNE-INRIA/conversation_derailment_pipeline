#!/bin/bash
#SBATCH --job-name=prepare_encoder_data
#SBATCH --output=logs/prepare_encoder_%j.out
#SBATCH --error=logs/prepare_encoder_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --account=marianne
#SBATCH --partition=cpucourt

set -euo pipefail

mkdir -p logs

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PREDICHATE
# Compute nodes have no internet access; rely on the login-node cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
encoder_python="${CONDA_PREFIX:?Conda activation did not set CONDA_PREFIX}/bin/python"
export PYTHONNOUSERSITE=1

if [[ ! -x "$encoder_python" ]]; then
  echo "Python is unavailable in encoder environment: $encoder_python" >&2
  exit 1
fi

"$encoder_python" - <<'PY'
import pandas
import torch
import transformers

print("Python environment: OK")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("pandas:", pandas.__version__)
PY

ROOT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$ROOT_DIR"

# Override with a valid local model directory if you have one.
DEBERTA_MODEL_NAME="${DEBERTA_MODEL_NAME:-microsoft/deberta-v3-base}"
ROBERTA_MODEL_NAME="${ROBERTA_MODEL_NAME:-roberta-base}"
echo "DeBERTa tokenizer/model: ${DEBERTA_MODEL_NAME}"
echo "RoBERTa tokenizer/model: ${ROBERTA_MODEL_NAME}"

"$encoder_python" scripts/prepare_encoder_pt.py \
  --train_csv ./data/train_samplesLLMs.csv \
  --val_csv ./data/val_samplesLLMs.csv \
  --test_csv ./data/test_samplesLLMs.csv \
  --output_dir $ROOT_DIR/encoder_data/deberta \
  --model_name "${DEBERTA_MODEL_NAME}" \
  --max_length 512

"$encoder_python" scripts/prepare_encoder_pt.py \
  --train_csv ./data/train_samplesLLMs.csv \
  --val_csv ./data/val_samplesLLMs.csv \
  --test_csv ./data/test_samplesLLMs.csv \
  --output_dir $ROOT_DIR/encoder_data/roberta \
  --model_name "${ROBERTA_MODEL_NAME}" \
  --max_length 512

python - <<'PY'
from pathlib import Path

expected = [
    Path("encoder_data/deberta/train_samples.pt"),
    Path("encoder_data/deberta/val_samples.pt"),
    Path("encoder_data/deberta/test_samples.pt"),
    Path("encoder_data/roberta/train_samples.pt"),
    Path("encoder_data/roberta/val_samples.pt"),
    Path("encoder_data/roberta/test_samples.pt"),
]
missing = [str(path) for path in expected if not path.is_file()]
if missing:
    raise SystemExit("Missing generated files: " + ", ".join(missing))

for path in expected:
    print(f"Created: {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
PY

echo "Encoder datasets prepared successfully."
