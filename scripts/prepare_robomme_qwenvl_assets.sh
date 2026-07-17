#!/usr/bin/env bash
set -euo pipefail

# Prepare raw RoboMME H5 data and the official QwenVL grounded subgoal adapter
# under HF_HOME-managed, repo-independent paths.

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_CLI="${HF_CLI:-huggingface-cli}"

ROBOMME_RAW_DATA_REPO="${ROBOMME_RAW_DATA_REPO:-Yinpei/robomme_data_h5}"
ROBOMME_RAW_DATA_DIR="${ROBOMME_RAW_DATA_DIR:-$HF_HOME/datasets/robomme_data_h5}"

ROBOMME_SUBGOAL_MODEL_REPO="${ROBOMME_SUBGOAL_MODEL_REPO:-Yinpei/vlm_subgoal_predictor}"
ROBOMME_SUBGOAL_ARCHIVE_SUBPATH="${ROBOMME_SUBGOAL_ARCHIVE_SUBPATH:-qwenvl/grounded_subgoal/checkpoint-1200.zip}"
ROBOMME_MODEL_ARCHIVE_DIR="${ROBOMME_MODEL_ARCHIVE_DIR:-$HF_HOME/downloads/vlm_subgoal_predictor}"
ROBOMME_MODEL_ROOT="${ROBOMME_MODEL_ROOT:-$HF_HOME/models/robomme}"
ROBOMME_GROUNDED_SUBGOAL_PARENT="${ROBOMME_GROUNDED_SUBGOAL_PARENT:-$ROBOMME_MODEL_ROOT/vlm_subgoal_predictor/qwenvl/grounded_subgoal}"
ROBOMME_GROUNDED_SUBGOAL_ADAPTER="${ROBOMME_GROUNDED_SUBGOAL_ADAPTER:-$ROBOMME_GROUNDED_SUBGOAL_PARENT/checkpoint-1200}"

SKIP_RAW_DATA=0
SKIP_MODEL=0
FORCE_UNZIP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-raw-data)
      SKIP_RAW_DATA=1
      shift
      ;;
    --skip-model)
      SKIP_MODEL=1
      shift
      ;;
    --force-unzip)
      FORCE_UNZIP=1
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--skip-raw-data] [--skip-model] [--force-unzip]

Environment variables:
  HF_HOME                         Default: $HOME/.cache/huggingface
  ROBOMME_RAW_DATA_REPO           Default: Yinpei/robomme_data_h5
  ROBOMME_RAW_DATA_DIR            Default: \$HF_HOME/datasets/robomme_data_h5
  ROBOMME_SUBGOAL_MODEL_REPO      Default: Yinpei/vlm_subgoal_predictor
  ROBOMME_MODEL_ARCHIVE_DIR       Default: \$HF_HOME/downloads/vlm_subgoal_predictor
  ROBOMME_MODEL_ROOT              Default: \$HF_HOME/models/robomme
  ROBOMME_GROUNDED_SUBGOAL_ADAPTER
                                  Default: \$ROBOMME_MODEL_ROOT/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

export HF_HOME

if ! command -v "$HF_CLI" >/dev/null 2>&1; then
  echo "Hugging Face CLI not found: $HF_CLI" >&2
  echo "Install/activate huggingface_hub, or set HF_CLI=/path/to/huggingface-cli." >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip command not found." >&2
  exit 1
fi

if [[ "$SKIP_RAW_DATA" -eq 0 ]]; then
  mkdir -p "$ROBOMME_RAW_DATA_DIR"
  echo "Downloading raw RoboMME H5 data to: $ROBOMME_RAW_DATA_DIR"
  "$HF_CLI" download "$ROBOMME_RAW_DATA_REPO" \
    --repo-type dataset \
    --local-dir "$ROBOMME_RAW_DATA_DIR"
fi

if [[ "$SKIP_MODEL" -eq 0 ]]; then
  mkdir -p "$ROBOMME_MODEL_ARCHIVE_DIR"
  echo "Downloading official QwenVL subgoal adapter archive to: $ROBOMME_MODEL_ARCHIVE_DIR"
  "$HF_CLI" download "$ROBOMME_SUBGOAL_MODEL_REPO" \
    "$ROBOMME_SUBGOAL_ARCHIVE_SUBPATH" \
    --local-dir "$ROBOMME_MODEL_ARCHIVE_DIR"

  archive_path="$ROBOMME_MODEL_ARCHIVE_DIR/$ROBOMME_SUBGOAL_ARCHIVE_SUBPATH"
  if [[ ! -f "$archive_path" ]]; then
    echo "Downloaded archive not found: $archive_path" >&2
    exit 1
  fi

  if [[ "$FORCE_UNZIP" -eq 1 || ! -f "$ROBOMME_GROUNDED_SUBGOAL_ADAPTER/adapter_config.json" ]]; then
    mkdir -p "$ROBOMME_GROUNDED_SUBGOAL_PARENT"
    echo "Extracting adapter to: $ROBOMME_GROUNDED_SUBGOAL_PARENT"
    unzip -q -o "$archive_path" -d "$ROBOMME_GROUNDED_SUBGOAL_PARENT"
  else
    echo "Adapter already exists: $ROBOMME_GROUNDED_SUBGOAL_ADAPTER"
  fi
fi

cat <<EOF

Prepared paths:
  HF_HOME=$HF_HOME
  ROBOMME_RAW_DATA_DIR=$ROBOMME_RAW_DATA_DIR
  ROBOMME_GROUNDED_SUBGOAL_ADAPTER=$ROBOMME_GROUNDED_SUBGOAL_ADAPTER
EOF
