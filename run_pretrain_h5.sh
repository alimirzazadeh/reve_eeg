#!/usr/bin/env bash
# Usage: bash run_pretrain_h5.sh [N_GPUS] [hydra overrides...]
set -euo pipefail

NGPU=${1:-1}
shift || true

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

mkdir -p /orcd/compute/dinaktbi/001/2026/EEG_FM/reve_pretrain_checkpoints

if [ "${NGPU}" -gt 1 ]; then
    conda run -n reve --no-capture-output \
        accelerate launch --num_processes "${NGPU}" \
            "${REPO_DIR}/src/pretrain_h5.py" trainer.n_gpus="${NGPU}" "$@"
else
    conda run -n reve --no-capture-output \
        python "${REPO_DIR}/src/pretrain_h5.py" trainer.n_gpus=1 "$@"
fi
