#!/bin/bash
#SBATCH --job-name=predichate_mistral
#SBATCH --output=logs/mistral_%j.out
#SBATCH --error=logs/mistral_%j.err
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

# Activate the environment created in the workspace.
# Example path expected by the cluster:

# If you created the environment by name instead, use:
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

# If any code uses a hardcoded device like cuda:0, replace it with:
# device = torch.device(f"cuda:{CUDA_VISIBLE_DEVICES}")
# or better: use device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$ROOT_DIR"

python scripts/train_llm_forecasting.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --train_path ./data/train_samplesLLMs.csv \
  --val_path ./data/val_samplesLLMs.csv \
  --output_dir ./checkpoints/mistral_7b \
  --epochs 4 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --lr 1e-4 \
  --max_length 8192 \
  --max_negative_alert_rate 0.10 \
  --patience 4 \
  --amp_dtype float32 \
  --seed 42