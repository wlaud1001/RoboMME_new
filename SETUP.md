# RoboMME_new Setup

This repository works best with three separate Python environments:

- training: Qwen3-VL, Swift, LoRA SFT, RLVR/GRPO
- evaluation: RoboMME simulator, dataset tooling, visualization, Qwen inference
- policy: official `openpi` low-level policy runtime

Do not merge all dependencies into one environment. The Qwen/Swift stack and
the official `openpi` stack pin different versions of core packages.

## 1. Repository And Paths

```bash
export ROBOMME_REPO="$(pwd)"
export ROBOMME_POLICY_REPO="$ROBOMME_REPO/official_baselines/robomme_policy_learning"
export ROBOMME_BENCH_REPO="$ROBOMME_REPO/external_dependencies/robomme_benchmark"
export HF_HOME="${HF_HOME:-/path/to/huggingface-cache}"

export ROBOMME_TRAIN_VENV="$ROBOMME_REPO/.venv-train"
export ROBOMME_EVAL_VENV="$ROBOMME_REPO/.venv-eval"
export ROBOMME_POLICY_VENV="$ROBOMME_REPO/.venv-policy"

cd "$ROBOMME_REPO"
git submodule update --init --recursive
```

Run that block from the repository root.

Path roles:

```text
$ROBOMME_REPO         top-level wrapper repo
$ROBOMME_POLICY_REPO  official_baselines/robomme_policy_learning
$ROBOMME_BENCH_REPO   external_dependencies/robomme_benchmark
```

Install `uv` if it is not available:

```bash
python -m pip install --user uv
```

## 2. Training Environment

Use this for Qwen3-VL, Swift, LoRA SFT, and RLVR/GRPO training. Do not install
the official `openpi` project here.

```bash
cd "$ROBOMME_REPO"

uv venv --python 3.11 "$ROBOMME_TRAIN_VENV"

UV_LINK_MODE=copy uv pip install \
  --python "$ROBOMME_TRAIN_VENV/bin/python" \
  -r "$ROBOMME_POLICY_REPO/examples/robomme/requirements.txt"

UV_LINK_MODE=copy uv pip install \
  --python "$ROBOMME_TRAIN_VENV/bin/python" \
  wandb

"$ROBOMME_TRAIN_VENV/bin/swift" sft --help | head
uv pip check --python "$ROBOMME_TRAIN_VENV/bin/python"
```

## 3. Evaluation Environment

Use this for dataset building, visualization, simulator execution, Qwen
inference, and agentic controller code. Do not install `openpi` here.

```bash
cd "$ROBOMME_REPO"

uv venv --python 3.11 "$ROBOMME_EVAL_VENV"

UV_LINK_MODE=copy uv pip install \
  --python "$ROBOMME_EVAL_VENV/bin/python" \
  -r "$ROBOMME_POLICY_REPO/examples/robomme/requirements.txt"

UV_LINK_MODE=copy uv pip install \
  --python "$ROBOMME_EVAL_VENV/bin/python" \
  -e "$ROBOMME_BENCH_REPO"

UV_LINK_MODE=copy uv pip install \
  --python "$ROBOMME_EVAL_VENV/bin/python" \
  -e "$ROBOMME_POLICY_REPO/packages/openpi-client"

UV_LINK_MODE=copy uv pip install \
  --python "$ROBOMME_EVAL_VENV/bin/python" \
  huggingface_hub tqdm pillow imageio imageio-ffmpeg h5py numpy opencv-python zstandard pyarrow

uv pip check --python "$ROBOMME_EVAL_VENV/bin/python"
```

The evaluation environment needs both the benchmark package and the policy repo
source tree on `PYTHONPATH` before running local scripts:

```bash
export PYTHONPATH="$ROBOMME_POLICY_REPO/src:$ROBOMME_BENCH_REPO/src:${PYTHONPATH:-}"
```

Why this matters:

- `examples/robomme/eval.py` imports `openpi_client`
- local scripts under `official_baselines/robomme_policy_learning` import from
  `src/openpi` and `src/mme_vla_suite`
- the simulator code imports from `robomme`

## 4. Policy Environment

Use this only for the official low-level `openpi` policy runtime. This
environment intentionally keeps the official dependency pins separate from
Qwen/Swift.

```bash
cd "$ROBOMME_REPO"

uv venv --python 3.11 "$ROBOMME_POLICY_VENV"

cd "$ROBOMME_POLICY_REPO"

GIT_LFS_SKIP_SMUDGE=1 \
UV_PROJECT_ENVIRONMENT="$ROBOMME_POLICY_VENV" \
UV_LINK_MODE=copy \
uv sync

UV_LINK_MODE=copy uv pip install \
  --python "$ROBOMME_POLICY_VENV/bin/python" \
  -e .

uv pip check --python "$ROBOMME_POLICY_VENV/bin/python"
```

## 5. Quick Checks

```bash
"$ROBOMME_TRAIN_VENV/bin/python" --version
"$ROBOMME_EVAL_VENV/bin/python" --version
"$ROBOMME_POLICY_VENV/bin/python" --version

"$ROBOMME_EVAL_VENV/bin/python" -c "import robomme; print('robomme ok')"
"$ROBOMME_EVAL_VENV/bin/python" -c "import openpi_client; print('openpi_client ok')"
"$ROBOMME_TRAIN_VENV/bin/python" -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
"$ROBOMME_POLICY_VENV/bin/python" -c "import openpi; import mme_vla_suite; print('policy imports ok')"
```

Optional lightweight simulator smoke test:

```bash
cd "$ROBOMME_POLICY_REPO"
export PYTHONPATH="$ROBOMME_POLICY_REPO/src:$ROBOMME_BENCH_REPO/src:${PYTHONPATH:-}"
"$ROBOMME_EVAL_VENV/bin/python" examples/robomme/simple_test.py
```

If you do not need artifacts from a smoke run, remove them afterward to keep the
repo clean.

## 6. Environment Roles

```text
.venv-train  = Qwen3-VL / Swift / LoRA and RLVR training
.venv-eval   = simulator, dataset build, visualization, Qwen inference
.venv-policy = official openpi low-level policy only
```

Typical working directories:

```text
Training / policy scripts:  cd $ROBOMME_POLICY_REPO
Benchmark package work:     cd $ROBOMME_BENCH_REPO
Top-level environment setup: cd $ROBOMME_REPO
```

Expected model/cache roots:

```text
$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/
$HF_HOME/models/robomme/mme_vla_suite/symbolic-grounded-subgoal/79999/
$HF_HOME/models/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200/
```
