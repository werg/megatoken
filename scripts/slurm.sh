#!/usr/bin/env bash
# Submit from an environment with the repository's tested container/runtime.
# Override the resource directives with sbatch flags for the actual cluster.
#SBATCH --job-name=megatoken
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=megatoken-%j.out
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_JOB_NODELIST:?Slurm did not supply the allocated nodes}"
MEGATOKEN_ROOT="${MEGATOKEN_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the shared repository directory}}"
cd -- "$MEGATOKEN_ROOT"
if [[ ! -f scripts/launch.sh ]]; then
  echo "Submit from the repository root, or set MEGATOKEN_ROOT to its shared absolute path" >&2
  exit 2
fi
MEGATOKEN_ROOT="$(pwd)"
node_list="$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
if [[ -z "$node_list" ]]; then
  echo "Unable to resolve the Slurm allocation" >&2
  exit 2
fi
mapfile -t allocated_nodes <<< "$node_list"
export NNODES="${SLURM_JOB_NUM_NODES:-${#allocated_nodes[@]}}"
export RDZV_ID="${RDZV_ID:-$SLURM_JOB_ID}"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-${allocated_nodes[0]}:${RDZV_PORT:-29500}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# One task per node is deliberate: each agent below spawns its own GPU workers.
# Atomic periodic checkpoints can be resumed after allocation termination.
exec srun --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --cpu-bind=none bash "$MEGATOKEN_ROOT/scripts/launch.sh" "$@"
