#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT="$REPO_ROOT/external_dependencies/Isaac-GR00T"

# Defaults for the absolute-joint, text-first RoboMME setup.
HF_HOME="${HF_HOME:-/data1/wlaud1001/huggingface}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
ROBOVLA_NUM_GPUS="${ROBOVLA_NUM_GPUS:-2}"
ROBOVLA_GLOBAL_BATCH_SIZE="${ROBOVLA_GLOBAL_BATCH_SIZE:-2}"
ROBOVLA_GRADIENT_ACCUMULATION_STEPS="${ROBOVLA_GRADIENT_ACCUMULATION_STEPS:-1}"
ROBOVLA_DATALOADER_WORKERS="${ROBOVLA_DATALOADER_WORKERS:-0}"
ROBOVLA_BASE_MODEL="${ROBOVLA_BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
ROBOVLA_LEARNING_RATE="${ROBOVLA_LEARNING_RATE:-0.00001}"
ROBOVLA_WEIGHT_DECAY="${ROBOVLA_WEIGHT_DECAY:-0.0001}"
ROBOVLA_WARMUP_RATIO="${ROBOVLA_WARMUP_RATIO:-0.05}"
ROBOVLA_MAX_STEPS="${ROBOVLA_MAX_STEPS:-10000}"
ROBOVLA_SAVE_STEPS="${ROBOVLA_SAVE_STEPS:-1000}"
ROBOVLA_SAVE_TOTAL_LIMIT="${ROBOVLA_SAVE_TOTAL_LIMIT:-5}"
ROBOVLA_OUTPUT_DIR="${ROBOVLA_OUTPUT_DIR:-$REPO_ROOT/runs/robovla/robomme_gr00t_abs_text_first}"
MASTER_PORT="${MASTER_PORT:-29501}"

export HF_HOME
export CUDA_VISIBLE_DEVICES
export CUDA_HOME
if [[ ! -x "$CUDA_HOME/bin/nvcc" && -x /usr/local/cuda-12.6/bin/nvcc ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
fi
export CUDA_INC_DIR="$CUDA_HOME/include"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT/src:$GR00T_ROOT:${PYTHONPATH:-}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv-train}"
export ROBOVLA_TRAIN_MODE="${ROBOVLA_TRAIN_MODE:-full}"

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

cd "$GR00T_ROOT"
ROBOMME_DATASET="${ROBOMME_DATASET:-$(resolve_hf_dataset_snapshot)}"

TRAIN_CMD=(uv run python "$REPO_ROOT/src/robovla/train_robomme.py")
if [[ "$ROBOVLA_NUM_GPUS" -gt 1 ]]; then
  TRAIN_CMD=(
    uv run torchrun
    --nproc_per_node="$ROBOVLA_NUM_GPUS"
    --master_port="$MASTER_PORT"
    "$REPO_ROOT/src/robovla/train_robomme.py"
  )
fi

"${TRAIN_CMD[@]}" \
  --base-model-path "$ROBOVLA_BASE_MODEL" \
  --dataset-path "$ROBOMME_DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$REPO_ROOT/src/robovla/configs/robomme_modality.py" \
  --output-dir "$ROBOVLA_OUTPUT_DIR" \
  --global-batch-size "$ROBOVLA_GLOBAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$ROBOVLA_GRADIENT_ACCUMULATION_STEPS" \
  --dataloader-num-workers "$ROBOVLA_DATALOADER_WORKERS" \
  --learning-rate "$ROBOVLA_LEARNING_RATE" \
  --weight-decay "$ROBOVLA_WEIGHT_DECAY" \
  --warmup-ratio "$ROBOVLA_WARMUP_RATIO" \
  --max-steps "$ROBOVLA_MAX_STEPS" \
  --save-steps "$ROBOVLA_SAVE_STEPS" \
  --save-total-limit "$ROBOVLA_SAVE_TOTAL_LIMIT" \
  --num-gpus "$ROBOVLA_NUM_GPUS" \
  "$@"
