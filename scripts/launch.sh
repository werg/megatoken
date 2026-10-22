#!/usr/bin/env bash
# One torchrun agent on this node; torchrun creates one worker per GPU.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: scripts/launch.sh CONFIG [training CLI arguments...]

Single node:
  NPROC_PER_NODE=8 scripts/launch.sh configs/train/coherence.example.yaml

Multiple nodes: run once per node with identical NNODES/RDZV_ID/RDZV_ENDPOINT,
and a distinct NODE_RANK in [0, NNODES). All paths are relative to the repository.
  NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 RDZV_ID=run-001 \
    RDZV_ENDPOINT=coordinator.example:29500 scripts/launch.sh configs/train/coherence.example.yaml

MEGATOKEN_PYTHON selects the Python executable. NPROC_PER_NODE defaults to gpu.
TORCHRUN_MAX_RESTARTS defaults to 0; explicit checkpoint resume preserves progress.
HELP
  exit 0
fi

MEGATOKEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$MEGATOKEN_ROOT"
if (( $# == 0 )) || [[ "$1" == -* ]]; then
  echo "Supply a prepared training config explicitly; there is no default" >&2
  exit 2
fi
CONFIG="$1"
shift
if [[ ! -f "$CONFIG" ]]; then
  echo "Training configuration does not exist: $CONFIG" >&2
  exit 2
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
MEGATOKEN_PYTHON="${MEGATOKEN_PYTHON:-python}"
if [[ ! "$NNODES" =~ ^[1-9][0-9]*$ || ! "$NODE_RANK" =~ ^[0-9]+$ ]]; then
  echo "NNODES must be positive and NODE_RANK must be a nonnegative integer" >&2
  exit 2
fi
if (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be smaller than NNODES" >&2
  exit 2
fi
if [[ "$NPROC_PER_NODE" != gpu && ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be gpu or a positive integer" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
launcher=("$MEGATOKEN_PYTHON" -m torch.distributed.run
          --nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE"
          --max-restarts "${TORCHRUN_MAX_RESTARTS:-0}")
if (( NNODES == 1 )); then
  launcher+=(--standalone)
else
  : "${RDZV_ENDPOINT:?Set RDZV_ENDPOINT to the reachable coordinator host:port}"
  : "${RDZV_ID:?Set the same unique RDZV_ID on every node in this training job}"
  launcher+=(--node-rank "$NODE_RANK" --rdzv-backend c10d
             --rdzv-endpoint "$RDZV_ENDPOINT" --rdzv-id "$RDZV_ID")
fi
exec "${launcher[@]}" -m megatoken train --config "$CONFIG" "$@"
