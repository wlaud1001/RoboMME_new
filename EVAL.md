# RoboMME Evaluation

This file is the practical evaluation guide for this repository.

Use it when you want to:

- evaluate a released MME-VLA checkpoint on RoboMME
- run a quick smoke evaluation before a full benchmark run
- use a checkpoint stored under `HF_HOME` instead of copying it into `runs/ckpts`
- implement GR00T eval without mixing up relative and absolute joint actions

This guide assumes you already followed [SETUP.md](./SETUP.md).

## GR00T Eval: Do Not Mix Delta And Absolute Joints

Current RoboMME GR00T training config uses:

```text
state.joint_position = absolute 7D joint state
action.joint_position = relative joint delta during training
action.gripper = absolute 1D gripper action
```

This is easy to misuse during eval. The rule is:

```text
RoboMME env expects absolute joint_angle actions: 7 joint angles + 1 gripper.
Do not pass raw GR00T model action_pred to RoboMME.
Do not manually add current_joint_state if using Gr00tPolicy.get_action().
```

Use GR00T's policy API for inference:

```python
gr00t_action, _ = gr00t_policy.get_action(gr00t_observation)
```

`Gr00tPolicy.get_action()` already calls the saved processor and decodes:

```text
normalized model output
-> unnormalized action
-> relative joint delta converted back to absolute joint position
```

So the RoboMME action conversion should only concatenate decoded outputs:

```python
joint = gr00t_action["joint_position"]  # already absolute, shape (B, T, 7)
gripper = gr00t_action["gripper"]       # absolute, shape (B, T, 1)
robomme_actions = np.concatenate([joint[0], gripper[0]], axis=-1)
```

Only add `current_joint_state` yourself if you deliberately bypass
`Gr00tPolicy.get_action()` and are holding a raw decoded delta. That should not
be the default eval path.

Local GR00T wrapper:

```text
src/robovla/robomme_gr00t_eval.py
```

One-episode smoke eval:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/data1/wlaud1001/huggingface \
MUJOCO_GL=egl \
.venv-train/bin/python scripts/eval_robovla_gr00t_one_episode.py \
  --model-path runs_b200/robovla/robomme_gr00t_full_ft/checkpoint-10000 \
  --task BinFill \
  --episode-id 0 \
  --max-steps 1 \
  --device cuda:0
```

Expected smoke output includes:

```text
action_chunk_shape=(16, 8)
reached max_steps=1 outcome=ongoing
```

## 1. Required Environments

Evaluation uses two separate environments:

- `.venv-policy`: serves the policy over websocket
- `.venv-eval`: runs the RoboMME simulator and eval client

Required shell variables:

```bash
export ROBOMME_REPO="$(pwd)"
export ROBOMME_POLICY_REPO="$ROBOMME_REPO/official_baselines/robomme_policy_learning"
export ROBOMME_BENCH_REPO="$ROBOMME_REPO/external_dependencies/robomme_benchmark"
export HF_HOME="${HF_HOME:-/path/to/huggingface-cache}"
export PYTHONPATH="$ROBOMME_POLICY_REPO/src:$ROBOMME_BENCH_REPO/src:${PYTHONPATH:-}"
```

Run that block from the repository root.

## 2. Required Assets

### RoboMME evaluation data

The simulator needs the RoboMME HDF5 dataset. In this machine it is available at:

```text
$HF_HOME/datasets/robomme_data_h5
```

### Policy checkpoint

Example released checkpoint:

```text
$HF_HOME/hub/models--Yinpei--mme_vla_suite/snapshots/5db4d53ddb98c7f80cab08792dd53d985d712ab1/perceptual-framesamp-modul
```

That directory contains:

```text
history_config.txt
79999.zip
```

The policy server expects an extracted checkpoint directory:

```text
.../perceptual-framesamp-modul/79999
```

## 3. Extract A Checkpoint From HF Cache

Do this once per checkpoint ID.

```bash
mkdir -p "$HF_HOME/hub/models--Yinpei--mme_vla_suite/snapshots/5db4d53ddb98c7f80cab08792dd53d985d712ab1/perceptual-framesamp-modul/79999"

"$ROBOMME_REPO/.venv-eval/bin/python" - <<'PY'
import os
from zipfile import ZipFile
from pathlib import Path

hf_home = Path(os.environ["HF_HOME"])
zip_path = (
    hf_home
    / "hub/models--Yinpei--mme_vla_suite/snapshots/"
    / "5db4d53ddb98c7f80cab08792dd53d985d712ab1/"
    / "perceptual-framesamp-modul/79999.zip"
)
out_dir = zip_path.parent / "79999"

with ZipFile(zip_path, "r") as zf:
    for member in zf.infolist():
        if member.filename.endswith("/"):
            continue
        p = Path(member.filename)
        parts = p.parts
        if "79999" in parts:
            idx = parts.index("79999")
            rel = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else Path(p.name)
        else:
            rel = Path(p.name)
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(dest, "wb") as dst:
            dst.write(src.read())

print(out_dir)
PY
```

This assumes the checkpoint zip already exists under
`$HF_HOME/hub/models--Yinpei--mme_vla_suite/snapshots/.../`.

Why not `unzip_ckpt.py` directly?

- Hugging Face snapshot files are symlinks into `hub/blobs/`
- the generic unzip script resolves the symlink target and may extract into the blob path instead of `79999/`

## 4. Smoke Evaluation

This is the fastest useful end-to-end check:

- one checkpoint
- one RoboMME task
- one episode
- shorter step limit

### Terminal 1: policy server

```bash
export ROBOMME_REPO="$(pwd)"
export ROBOMME_POLICY_REPO="$ROBOMME_REPO/official_baselines/robomme_policy_learning"
export ROBOMME_BENCH_REPO="$ROBOMME_REPO/external_dependencies/robomme_benchmark"
export HF_HOME="${HF_HOME:-/path/to/huggingface-cache}"
export PYTHONPATH="$ROBOMME_POLICY_REPO/src:$ROBOMME_BENCH_REPO/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

cd "$ROBOMME_POLICY_REPO"

"$ROBOMME_REPO/.venv-policy/bin/python" scripts/serve_policy.py \
  --seed=7 \
  --port=8013 \
  policy:checkpoint \
  --policy.dir="$HF_HOME/hub/models--Yinpei--mme_vla_suite/snapshots/5db4d53ddb98c7f80cab08792dd53d985d712ab1/perceptual-framesamp-modul/79999" \
  --policy.config=mme_vla_suite
```

### Terminal 2: eval client

```bash
export ROBOMME_REPO="$(pwd)"
export ROBOMME_POLICY_REPO="$ROBOMME_REPO/official_baselines/robomme_policy_learning"
export ROBOMME_BENCH_REPO="$ROBOMME_REPO/external_dependencies/robomme_benchmark"
export HF_HOME="${HF_HOME:-/path/to/huggingface-cache}"
export PYTHONPATH="$ROBOMME_POLICY_REPO/src:$ROBOMME_BENCH_REPO/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=osmesa
export SAPIEN_RENDER_DEVICE=cpu

cd "$ROBOMME_POLICY_REPO/examples/robomme"

"$ROBOMME_REPO/.venv-eval/bin/python" eval.py \
  --args.model_seed=7 \
  --args.port=8013 \
  --args.policy_name=perceptual-framesamp-modul \
  --args.model_ckpt_id=79999 \
  --args.only_tasks=BinFill \
  --args.max_steps=120 \
  --args.save_dir=../../runs/_smoke_eval
```

Important:

- `max_steps=120` is for smoke only
- default full-eval behavior is much longer; the code default is `1300`

Smoke outputs will be written under:

```text
official_baselines/robomme_policy_learning/runs/_smoke_eval/
```

## 5. Full Evaluation

For a normal run, use the same two-terminal pattern but raise the step limit:

```bash
--args.max_steps=1300
```

Example:

### Terminal 1

```bash
export ROBOMME_REPO="$(pwd)"
export ROBOMME_POLICY_REPO="$ROBOMME_REPO/official_baselines/robomme_policy_learning"
export ROBOMME_BENCH_REPO="$ROBOMME_REPO/external_dependencies/robomme_benchmark"
export HF_HOME="${HF_HOME:-/path/to/huggingface-cache}"
export PYTHONPATH="$ROBOMME_POLICY_REPO/src:$ROBOMME_BENCH_REPO/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

cd "$ROBOMME_POLICY_REPO"

"$ROBOMME_REPO/.venv-policy/bin/python" scripts/serve_policy.py \
  --seed=7 \
  --port=8013 \
  policy:checkpoint \
  --policy.dir="$HF_HOME/hub/models--Yinpei--mme_vla_suite/snapshots/5db4d53ddb98c7f80cab08792dd53d985d712ab1/perceptual-framesamp-modul/79999" \
  --policy.config=mme_vla_suite
```

### Terminal 2

```bash
export ROBOMME_REPO="$(pwd)"
export ROBOMME_POLICY_REPO="$ROBOMME_REPO/official_baselines/robomme_policy_learning"
export ROBOMME_BENCH_REPO="$ROBOMME_REPO/external_dependencies/robomme_benchmark"
export HF_HOME="${HF_HOME:-/path/to/huggingface-cache}"
export PYTHONPATH="$ROBOMME_POLICY_REPO/src:$ROBOMME_BENCH_REPO/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=osmesa
export SAPIEN_RENDER_DEVICE=cpu

cd "$ROBOMME_POLICY_REPO/examples/robomme"

"$ROBOMME_REPO/.venv-eval/bin/python" eval.py \
  --args.model_seed=7 \
  --args.port=8013 \
  --args.policy_name=perceptual-framesamp-modul \
  --args.model_ckpt_id=79999 \
  --args.only_tasks=BinFill \
  --args.max_steps=1300
```

If you want the full benchmark instead of one task, remove:

```bash
--args.only_tasks=BinFill
```

## 6. Output Layout

Evaluation writes to:

```text
official_baselines/robomme_policy_learning/runs/evaluation/<policy_name>/ckpt<id>/seed<seed>/
```

Common files:

- `progress.json`: episode-by-episode progress
- `log.json`: final success rates
- `videos/*.mp4`: rollout videos

## 7. Known Practical Issues

### GPU selection

JAX may grab the wrong GPU unless you pin it explicitly.

Use:

```bash
export CUDA_VISIBLE_DEVICES=0
```

for the policy server.

### Headless rendering

For remote or non-display evaluation:

```bash
export MUJOCO_GL=osmesa
export SAPIEN_RENDER_DEVICE=cpu
```

### Relative config path

Run `scripts/serve_policy.py` from:

```text
$ROBOMME_POLICY_REPO
```

The history config loader uses a repo-relative path and may fail if you launch
it from the wrong working directory.

### Checkpoint format

The server expects:

```text
79999/
  assets/
  params/
```

It cannot serve directly from `79999.zip`.

## 8. Existing Upstream Docs

The upstream project also has:

- [official_baselines/robomme_policy_learning/README.md](./official_baselines/robomme_policy_learning/README.md)
- [official_baselines/robomme_policy_learning/docs/manual_evaluation.md](./official_baselines/robomme_policy_learning/docs/manual_evaluation.md)

Use those for model-by-model command variants.
Use this file for the actual execution procedure in this repository.
