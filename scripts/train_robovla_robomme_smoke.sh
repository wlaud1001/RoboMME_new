#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT="$REPO_ROOT/external_dependencies/Isaac-GR00T"
ROBOMME_DATASET="${ROBOMME_DATASET:-/data1/wlaud1001/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/snapshots/1510653cccb4d9e5165fb3141c06d88053decc20}"

export HF_HOME="${HF_HOME:-/data1/wlaud1001/huggingface}"
export PYTHONPATH="$REPO_ROOT/src:$GR00T_ROOT:${PYTHONPATH:-}"

cd "$GR00T_ROOT"

uv run python "$REPO_ROOT/src/robovla/train_robomme.py" \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$ROBOMME_DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$REPO_ROOT/src/robovla/configs/robomme_modality.py" \
  --output-dir "$REPO_ROOT/runs/robovla/robomme_smoke" \
  --global-batch-size 1 \
  --dataloader-num-workers 0 \
  --max-steps 10 \
  --save-steps 10 \
  --save-total-limit 1 \
  --num-gpus 1 \
  "$@"
