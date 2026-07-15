#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT="$REPO_ROOT/external_dependencies/Isaac-GR00T"
export HF_HOME="${HF_HOME:-/data1/wlaud1001/huggingface}"

resolve_hf_dataset_snapshot() {
  uv run python - <<'PY'
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id="Yinpei/robomme_data_lerobot",
        repo_type="dataset",
        local_files_only=True,
    )
)
PY
}

ROBOMME_SMOKE_EPISODES="${ROBOMME_SMOKE_EPISODES:-8}"
ROBOMME_DATASET="${ROBOMME_DATASET:-$REPO_ROOT/runs/_smoke/robomme_lerobot_${ROBOMME_SMOKE_EPISODES}eps}"
ROBOVLA_NUM_GPUS="${ROBOVLA_NUM_GPUS:-2}"
ROBOVLA_GLOBAL_BATCH_SIZE="${ROBOVLA_GLOBAL_BATCH_SIZE:-2}"
export ROBOVLA_TRAIN_MODE="${ROBOVLA_TRAIN_MODE:-action_head}"

export PYTHONPATH="$REPO_ROOT/src:$GR00T_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" && -x /usr/local/cuda-12.6/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
fi
export CUDA_INC_DIR="$CUDA_HOME/include"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-train"

cd "$GR00T_ROOT"
ROBOMME_SOURCE_DATASET="${ROBOMME_SOURCE_DATASET:-$(resolve_hf_dataset_snapshot)}"

uv run python - <<PY
from robovla.robomme_lerobot import make_robomme_lerobot_subset

info = make_robomme_lerobot_subset(
    source_path="$ROBOMME_SOURCE_DATASET",
    output_path="$ROBOMME_DATASET",
    num_episodes=int("$ROBOMME_SMOKE_EPISODES"),
)
print(f"Using smoke dataset: {info.root} ({info.total_episodes} episodes)")
PY

TRAIN_CMD=(uv run python "$REPO_ROOT/src/robovla/train_robomme.py")
if [[ "$ROBOVLA_NUM_GPUS" -gt 1 ]]; then
  TRAIN_CMD=(
    uv run torchrun
    --nproc_per_node="$ROBOVLA_NUM_GPUS"
    --master_port="${MASTER_PORT:-29501}"
    "$REPO_ROOT/src/robovla/train_robomme.py"
  )
fi

"${TRAIN_CMD[@]}" \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$ROBOMME_DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$REPO_ROOT/src/robovla/configs/robomme_modality.py" \
  --output-dir "$REPO_ROOT/runs/robovla/robomme_smoke" \
  --global-batch-size "$ROBOVLA_GLOBAL_BATCH_SIZE" \
  --dataloader-num-workers 0 \
  --max-steps 10 \
  --save-steps 10 \
  --save-total-limit 1 \
  --num-gpus "$ROBOVLA_NUM_GPUS" \
  "$@"
