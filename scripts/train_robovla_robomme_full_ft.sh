#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT="$REPO_ROOT/external_dependencies/Isaac-GR00T"

# Edit these defaults here, or override them from the command line.
HF_HOME="/data1/wlaud1001/huggingface"
CUDA_VISIBLE_DEVICES="0,1"
CUDA_HOME="/usr/local/cuda-12.6"
ROBOVLA_NUM_GPUS=2
ROBOVLA_GLOBAL_BATCH_SIZE=2
ROBOVLA_DATALOADER_WORKERS=0
ROBOVLA_BASE_MODEL="nvidia/GR00T-N1.7-3B"
ROBOVLA_MAX_STEPS=10000
ROBOVLA_SAVE_STEPS=1000
ROBOVLA_SAVE_TOTAL_LIMIT=5
ROBOVLA_OUTPUT_DIR="$REPO_ROOT/runs/robovla/robomme_gr00t_full_ft"
MASTER_PORT=29501

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
export ROBOVLA_TRAIN_MODE=full

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
  --dataloader-num-workers "$ROBOVLA_DATALOADER_WORKERS" \
  --max-steps "$ROBOVLA_MAX_STEPS" \
  --save-steps "$ROBOVLA_SAVE_STEPS" \
  --save-total-limit "$ROBOVLA_SAVE_TOTAL_LIMIT" \
  --num-gpus "$ROBOVLA_NUM_GPUS" \
  "$@"
