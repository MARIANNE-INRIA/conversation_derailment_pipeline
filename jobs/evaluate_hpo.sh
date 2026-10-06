#!/bin/bash
#SBATCH --job-name=predichate_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-12:00:00
#SBATCH --account=marianne
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1

set -euo pipefail
hpo_default_env="PREDICHATE"
# DeBERTa needs torch>=2.6; select the encoders env when --output_dir points at a deberta search.
if [[ "$*" == *deberta* ]]; then
  hpo_default_env="PREDICHATE_ENCODERS"
fi
hpo_conda_env="${HPO_CONDA_PREFIX:-$hpo_default_env}"
module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$hpo_conda_env"
# Compute nodes have no internet access; rely on the login-node cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
hash -r
hpo_python="${CONDA_PREFIX:?Conda activation did not set CONDA_PREFIX}/bin/python"
if [[ ! -x "$hpo_python" ]]; then
  echo "Python is unavailable on this node: $hpo_python" >&2
  exit 1
fi
printf 'Requested environment: %s\nConda prefix: %s\nPython executable: %s\n' "$hpo_conda_env" "$CONDA_PREFIX" "$hpo_python"
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
"$hpo_python" -u scripts/evaluate_hpo.py "$@"
