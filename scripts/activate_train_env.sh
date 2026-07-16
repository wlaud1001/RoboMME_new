#!/usr/bin/env bash

_robomme_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBOMME_REPO="$(cd "$_robomme_script_dir/.." && pwd)"
unset _robomme_script_dir

export ROBOMME_POLICY_REPO="$ROBOMME_REPO/official_baselines/robomme_policy_learning"
export ROBOMME_BENCH_REPO="$ROBOMME_REPO/external_dependencies/robomme_benchmark"
export ROBOMME_GR00T_REPO="$ROBOMME_REPO/external_dependencies/Isaac-GR00T"
export ROBOMME_TRAIN_VENV="$ROBOMME_REPO/.venv-train"

export HF_HOME="/NHNHOME/WORKSPACE/0526050006_AA/wlaud1001/huggingface"
export CUDA_VISIBLE_DEVICES="0,1"
export NO_ALBUMENTATIONS_UPDATE="1"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      export CUDA_HOME="$candidate"
      break
    fi
  done
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME="/usr/local/cuda"
fi
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_INC_DIR="$CUDA_HOME/include"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

export VIRTUAL_ENV="$ROBOMME_TRAIN_VENV"
case ":$PATH:" in
  *":$ROBOMME_TRAIN_VENV/bin:"*) ;;
  *) export PATH="$ROBOMME_TRAIN_VENV/bin:$PATH" ;;
esac

case ":${PYTHONPATH:-}:" in
  *":$ROBOMME_REPO/src:"*) ;;
  *) export PYTHONPATH="$ROBOMME_REPO/src:$ROBOMME_GR00T_REPO:$ROBOMME_POLICY_REPO/src:$ROBOMME_BENCH_REPO/src:${PYTHONPATH:-}" ;;
esac
