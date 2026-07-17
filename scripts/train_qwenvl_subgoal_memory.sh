#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_REPO="$REPO_ROOT/official_baselines/robomme_policy_learning"

# Paths
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
QWENVL_DATASET_PATH="${QWENVL_DATASET_PATH:-$REPO_ROOT/outputs/robomme_qwenvl_memory_h16_k16_swift/grounded_subgoal_train.jsonl}"
QWENVL_OUTPUT_DIR="${QWENVL_OUTPUT_DIR:-$REPO_ROOT/runs/qwenvl_subgoal_memory_h16_k16/grounded_subgoal}"
SWIFT_BIN="${SWIFT_BIN:-swift}"

# Start from the official RoboMME grounded QwenVL subgoal predictor LoRA adapter.
# Set QWENVL_INIT_ADAPTER="" to train from the base Qwen3-VL model instead.
ROBOMME_MODEL_ROOT="${ROBOMME_MODEL_ROOT:-$HF_HOME/models/robomme}"
QWENVL_INIT_ADAPTER="${QWENVL_INIT_ADAPTER:-$ROBOMME_MODEL_ROOT/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200}"
QWENVL_AUTO_PREPARE_ADAPTER="${QWENVL_AUTO_PREPARE_ADAPTER:-true}"

# Launch
QWENVL_NPROC_PER_NODE="${QWENVL_NPROC_PER_NODE:-}"
MASTER_PORT="${MASTER_PORT:-29511}"

# Visual token budget for our mixed-resolution dataset:
# past/demo are 128x128, current is 256x256.
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-64}"
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-16}"
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-10}"

# Training knobs
QWENVL_MODEL="${QWENVL_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
QWENVL_USE_HF="${QWENVL_USE_HF:-true}"
QWENVL_TRAIN_TYPE="${QWENVL_TRAIN_TYPE:-lora}"
QWENVL_TORCH_DTYPE="${QWENVL_TORCH_DTYPE:-bfloat16}"
QWENVL_NUM_TRAIN_EPOCHS="${QWENVL_NUM_TRAIN_EPOCHS:-2}"
QWENVL_MAX_STEPS="${QWENVL_MAX_STEPS:-}"
QWENVL_PER_DEVICE_BATCH_SIZE="${QWENVL_PER_DEVICE_BATCH_SIZE:-4}"
QWENVL_GRADIENT_ACCUMULATION_STEPS="${QWENVL_GRADIENT_ACCUMULATION_STEPS:-4}"
QWENVL_LEARNING_RATE="${QWENVL_LEARNING_RATE:-1e-4}"
QWENVL_WARMUP_RATIO="${QWENVL_WARMUP_RATIO:-0.05}"
QWENVL_LORA_RANK="${QWENVL_LORA_RANK:-16}"
QWENVL_LORA_ALPHA="${QWENVL_LORA_ALPHA:-32}"
QWENVL_TARGET_MODULES="${QWENVL_TARGET_MODULES:-all-linear}"
QWENVL_MAX_LENGTH="${QWENVL_MAX_LENGTH:-3200}"
QWENVL_SAVE_STEPS="${QWENVL_SAVE_STEPS:-100}"
QWENVL_SAVE_TOTAL_LIMIT="${QWENVL_SAVE_TOTAL_LIMIT:-2}"
QWENVL_LOGGING_STEPS="${QWENVL_LOGGING_STEPS:-100}"
QWENVL_DATASET_NUM_PROC="${QWENVL_DATASET_NUM_PROC:-4}"
QWENVL_DATALOADER_WORKERS="${QWENVL_DATALOADER_WORKERS:-4}"
QWENVL_DEEPSPEED="${QWENVL_DEEPSPEED:-zero2}"
QWENVL_ATTN_IMPL="${QWENVL_ATTN_IMPL:-sdpa}"
QWENVL_FREEZE_VIT="${QWENVL_FREEZE_VIT:-true}"
QWENVL_FREEZE_ALIGNER="${QWENVL_FREEZE_ALIGNER:-true}"
QWENVL_GRADIENT_CHECKPOINTING="${QWENVL_GRADIENT_CHECKPOINTING:-true}"
QWENVL_VIT_GRADIENT_CHECKPOINTING="${QWENVL_VIT_GRADIENT_CHECKPOINTING:-false}"
QWENVL_LOAD_FROM_CACHE_FILE="${QWENVL_LOAD_FROM_CACHE_FILE:-true}"

export HF_HOME
export IMAGE_MAX_TOKEN_NUM
export VIDEO_MAX_TOKEN_NUM
export FPS_MAX_FRAMES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-$PYTORCH_CUDA_ALLOC_CONF}"
export MASTER_PORT

if [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    export CUDA_HOME
  else
    shopt -s nullglob
    _cuda_nvcc_candidates=(/usr/local/cuda*/bin/nvcc)
    shopt -u nullglob
    if [[ "${#_cuda_nvcc_candidates[@]}" -gt 0 ]]; then
      IFS=$'\n' _cuda_nvcc_candidates=($(printf '%s\n' "${_cuda_nvcc_candidates[@]}" | sort -V))
      _cuda_nvcc_last=$((${#_cuda_nvcc_candidates[@]} - 1))
      CUDA_HOME="$(cd "$(dirname "${_cuda_nvcc_candidates[$_cuda_nvcc_last]}")/.." && pwd)"
      export CUDA_HOME
      export PATH="$CUDA_HOME/bin:$PATH"
    fi
    unset _cuda_nvcc_candidates _cuda_nvcc_last
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
fi

if [[ -z "$QWENVL_NPROC_PER_NODE" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _qwen_devices <<< "$CUDA_VISIBLE_DEVICES"
    QWENVL_NPROC_PER_NODE="${#_qwen_devices[@]}"
    unset _qwen_devices
  else
    QWENVL_NPROC_PER_NODE=1
  fi
fi
export NPROC_PER_NODE="$QWENVL_NPROC_PER_NODE"

if [[ ! -f "$QWENVL_DATASET_PATH" ]]; then
  echo "Dataset JSONL not found: $QWENVL_DATASET_PATH" >&2
  exit 1
fi
QWENVL_DATASET_PATH="$(realpath "$QWENVL_DATASET_PATH")"

if [[ -n "$QWENVL_INIT_ADAPTER" && ! -d "$QWENVL_INIT_ADAPTER" ]]; then
  if [[ "$QWENVL_AUTO_PREPARE_ADAPTER" == "true" && -x "$REPO_ROOT/scripts/prepare_robomme_qwenvl_assets.sh" ]]; then
    echo "Initial adapter not found. Preparing official adapter under HF_HOME: $QWENVL_INIT_ADAPTER"
    HF_HOME="$HF_HOME" \
      ROBOMME_MODEL_ROOT="$ROBOMME_MODEL_ROOT" \
      ROBOMME_GROUNDED_SUBGOAL_PARENT="$(dirname "$QWENVL_INIT_ADAPTER")" \
      ROBOMME_GROUNDED_SUBGOAL_ADAPTER="$QWENVL_INIT_ADAPTER" \
      "$REPO_ROOT/scripts/prepare_robomme_qwenvl_assets.sh" --skip-raw-data
  fi

  if [[ ! -d "$QWENVL_INIT_ADAPTER" ]]; then
    echo "Initial adapter directory not found: $QWENVL_INIT_ADAPTER" >&2
    echo "Run scripts/prepare_robomme_qwenvl_assets.sh, or set QWENVL_INIT_ADAPTER to an existing adapter path." >&2
    exit 1
  fi
  QWENVL_INIT_ADAPTER="$(realpath "$QWENVL_INIT_ADAPTER")"
fi

if ! command -v "$SWIFT_BIN" >/dev/null 2>&1; then
  echo "swift executable not found: $SWIFT_BIN" >&2
  echo "Activate the environment that has ms-swift installed, or set SWIFT_BIN=/path/to/swift." >&2
  exit 1
fi

mkdir -p "$QWENVL_OUTPUT_DIR"
QWENVL_OUTPUT_DIR="$(realpath "$QWENVL_OUTPUT_DIR")"

TRAIN_CMD=(
  "$SWIFT_BIN" sft
  --model "$QWENVL_MODEL"
  --use_hf "$QWENVL_USE_HF"
  --dataset "$QWENVL_DATASET_PATH"
  --split_dataset_ratio 0.0
  --load_from_cache_file "$QWENVL_LOAD_FROM_CACHE_FILE"
  --packing false
  --train_type "$QWENVL_TRAIN_TYPE"
  --torch_dtype "$QWENVL_TORCH_DTYPE"
  --num_train_epochs "$QWENVL_NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$QWENVL_PER_DEVICE_BATCH_SIZE"
  --gradient_accumulation_steps "$QWENVL_GRADIENT_ACCUMULATION_STEPS"
  --attn_impl "$QWENVL_ATTN_IMPL"
  --padding_free false
  --learning_rate "$QWENVL_LEARNING_RATE"
  --lora_rank "$QWENVL_LORA_RANK"
  --lora_alpha "$QWENVL_LORA_ALPHA"
  --target_modules "$QWENVL_TARGET_MODULES"
  --freeze_vit "$QWENVL_FREEZE_VIT"
  --freeze_aligner "$QWENVL_FREEZE_ALIGNER"
  --gradient_checkpointing "$QWENVL_GRADIENT_CHECKPOINTING"
  --vit_gradient_checkpointing "$QWENVL_VIT_GRADIENT_CHECKPOINTING"
  --save_steps "$QWENVL_SAVE_STEPS"
  --save_total_limit "$QWENVL_SAVE_TOTAL_LIMIT"
  --logging_steps "$QWENVL_LOGGING_STEPS"
  --max_length "$QWENVL_MAX_LENGTH"
  --output_dir "$QWENVL_OUTPUT_DIR"
  --warmup_ratio "$QWENVL_WARMUP_RATIO"
  --deepspeed "$QWENVL_DEEPSPEED"
  --dataset_num_proc "$QWENVL_DATASET_NUM_PROC"
  --dataloader_num_workers "$QWENVL_DATALOADER_WORKERS"
)

if [[ -n "$QWENVL_INIT_ADAPTER" ]]; then
  TRAIN_CMD+=(--adapters "$QWENVL_INIT_ADAPTER")
fi

if [[ -n "$QWENVL_MAX_STEPS" ]]; then
  TRAIN_CMD+=(--max_steps "$QWENVL_MAX_STEPS")
fi

cd "$POLICY_REPO"
printf 'Running command:\n'
printf '  %q' "${TRAIN_CMD[@]}"
printf '\n'

"${TRAIN_CMD[@]}" "$@"
