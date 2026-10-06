#!/bin/bash
#SBATCH --job-name=predichate_hpo
#SBATCH --output=logs/hpo_%j.out
#SBATCH --error=logs/hpo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-12:00:00
#SBATCH --account=marianne
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
hpo_pid=""
hpo_pause_requested=0
hpo_terminated=0
forward_checkpoint_signal() {
  hpo_pause_requested=1
  hpo_wait_interrupted=1
  if [[ -n "$hpo_pid" ]]; then
    kill -USR1 "$hpo_pid" 2>/dev/null || true
  fi
}
trap forward_checkpoint_signal USR1
trap 'hpo_terminated=1; forward_checkpoint_signal' TERM
# DeBERTa needs torch>=2.6 for its own trainer path; default it to PREDICHATE_ENCODERS.
hpo_default_env="PREDICHATE"
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--config" ]]; then
    j=$((i + 1))
    if [[ "${!j:-}" == *deberta* ]]; then
      hpo_default_env="PREDICHATE_ENCODERS"
    fi
    break
  fi
done
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
"$hpo_python" -c 'import sys; print("Python:", sys.executable, flush=True); print("Environment:", sys.prefix, flush=True); import torch; assert torch.cuda.is_available(), "No CUDA GPU allocated"; print(torch.cuda.get_device_name(0))' || {
  echo "PyTorch is missing or CUDA is unavailable in $hpo_conda_env." >&2
  echo "Rebuild it with: bash scripts/rebuild_predicate_env.sh $hpo_conda_env" >&2
  exit 1
}
# Arguments are forwarded unchanged; create logs/ BEFORE calling sbatch.
if (( hpo_pause_requested )); then
  hpo_status=75
else
  "$hpo_python" -u scripts/run_hpo.py "$@" &
  hpo_pid=$!
  # A trapped signal interrupts bash's wait before Python has saved its checkpoint.
  # Wait again until the orchestrator actually exits.
  while true; do
    hpo_wait_interrupted=0
    set +e
    wait "$hpo_pid"
    hpo_status=$?
    set -e
    if (( hpo_wait_interrupted )); then
      continue
    fi
    if ! kill -0 "$hpo_pid" 2>/dev/null; then
      break
    fi
  done
  hpo_pid=""
fi
if [[ "$hpo_status" -eq 75 ]]; then
  echo "Checkpoint pause: resubmit this command to resume the same training run."
  if [[ "${HPO_AUTO_REQUEUE:-0}" == 1 && "$hpo_terminated" == 0 ]]; then
    if (( ${SLURM_RESTART_COUNT:-0} < ${HPO_MAX_RESTARTS:-10} )); then
      echo "Requesting Slurm requeue for job ${SLURM_JOB_ID}."
      scontrol requeue "$SLURM_JOB_ID"
    else
      echo "Automatic requeue limit reached (${HPO_MAX_RESTARTS:-10}); resume manually."
    fi
  fi
fi
exit "$hpo_status"
