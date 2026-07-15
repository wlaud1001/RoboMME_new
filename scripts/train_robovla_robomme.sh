#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT="$REPO_ROOT/external_dependencies/Isaac-GR00T"

export HF_HOME="${HF_HOME:-/data1/wlaud1001/huggingface}"
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

ROBOVLA_NUM_GPUS="${ROBOVLA_NUM_GPUS:-2}"
ROBOVLA_GLOBAL_BATCH_SIZE="${ROBOVLA_GLOBAL_BATCH_SIZE:-2}"
export ROBOVLA_TRAIN_MODE="${ROBOVLA_TRAIN_MODE:-action_head}"

cd "$GR00T_ROOT"
ROBOMME_DATASET="${ROBOMME_DATASET:-$(resolve_hf_dataset_snapshot)}"

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
  --base-model-path "${ROBOVLA_BASE_MODEL:-nvidia/GR00T-N1.7-3B}" \
  --dataset-path "$ROBOMME_DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$REPO_ROOT/src/robovla/configs/robomme_modality.py" \
  --output-dir "$REPO_ROOT/runs/robovla/robomme_gr00t_n1d7_full" \
  --global-batch-size "$ROBOVLA_GLOBAL_BATCH_SIZE" \
  --dataloader-num-workers "${ROBOVLA_DATALOADER_WORKERS:-0}" \
  --max-steps 10000 \
  --save-steps 1000 \
  --save-total-limit 5 \
  --num-gpus "$ROBOVLA_NUM_GPUS" \
  "$@"
