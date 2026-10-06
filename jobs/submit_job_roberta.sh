#!/bin/bash
#SBATCH --job-name=predichate_roberta
#SBATCH --output=logs/roberta_%j.out
#SBATCH --error=logs/roberta_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-12:00:00
#SBATCH --account=marianne
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1

set -euo pipefail
mkdir -p logs

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PREDICHATE
# Compute nodes have no internet access; rely on the login-node cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is available in this Slurm allocation.")
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
PY

ROOT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$ROOT_DIR"

ROBERTA_MODEL_NAME="${ROBERTA_MODEL_NAME:-roberta-base}"
echo "RoBERTa model: ${ROBERTA_MODEL_NAME}"

python scripts/train_roberta.py \
  --train_path $ROOT_DIR/encoder_data/roberta/train_samples.pt \
  --val_path $ROOT_DIR/encoder_data/roberta/val_samples.pt \
  --model_name "${ROBERTA_MODEL_NAME}" \
  --output_dir ./checkpoints/roberta \
  --epochs 4 \
  --batch_size 4 \
  --lr 6.7e-5 \
  --patience 3 \
  --seed 42
