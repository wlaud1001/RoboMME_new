# Training GR00T on RoboMME

This repo keeps the upstream GR00T code in `external_dependencies/Isaac-GR00T`
unchanged. Local RoboMME-specific training glue lives under `src/robovla`.

## What This Uses

- Base model: `nvidia/GR00T-N1.7-3B`
- Dataset: `Yinpei/robomme_data_lerobot`
- Local dataset cache:
  `/data1/wlaud1001/huggingface/hub/datasets--Yinpei--robomme_data_lerobot/snapshots/1510653cccb4d9e5165fb3141c06d88053decc20`
- GPUs: use only GPU `0,1`
- HF cache: `HF_HOME=/data1/wlaud1001/huggingface`

## Local Wrapper Files

- `src/robovla/configs/robomme_modality.py`
  - Registers RoboMME as `EmbodimentTag.NEW_EMBODIMENT`.
  - Maps `image`, `wrist_image`, 7D `joint_position`, and 1D `gripper`.
  - Uses relative joint actions and absolute gripper actions.

- `src/robovla/robomme_lerobot.py`
  - Creates the missing GR00T `meta/modality.json`.
  - Converts RoboMME inline parquet image bytes into GR00T `video.*` fields.
  - Can create a small symlinked smoke dataset under `runs/_smoke/`.

- `src/robovla/train_robomme.py`
  - Loads the modality config.
  - Installs the RoboMME dataset adapter without modifying GR00T.
  - Calls the normal GR00T finetune pipeline.

## Quick Smoke Test

Use the wrapper script:

```bash
./scripts/train_robovla_robomme_smoke.sh --max-steps 1 --save-steps 1
```

By default this:

- Uses `CUDA_VISIBLE_DEVICES=0,1`
- Uses `torchrun --nproc_per_node=2`
- Uses `ROBOVLA_TRAIN_MODE=action_head`
- Creates an 8-episode smoke dataset at `runs/_smoke/robomme_lerobot_8eps`
- Writes outputs to `runs/robovla/robomme_smoke` unless overridden

For an explicit output directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
ROBOMME_SMOKE_EPISODES=8 \
ROBOVLA_NUM_GPUS=2 \
ROBOVLA_GLOBAL_BATCH_SIZE=2 \
./scripts/train_robovla_robomme_smoke.sh \
  --max-steps 1 \
  --save-steps 1 \
  --output-dir runs/_smoke/gr00t_robomme_2gpu_1step
```

## Why Smoke Uses a Subset

GR00T generates `meta/stats.json` and `meta/relative_stats.json` before
training. On the full RoboMME dataset this reads all 1600 parquet files on CPU,
so GPU usage stays at zero for a while.

For quick validation, the smoke script creates a small symlinked dataset. This
keeps stats generation short and lets us reach the actual GPU training step.

## Multi-GPU Requirement

Do not rely on `--num-gpus 2` alone. GR00T expects the process launcher world
size to match `--num-gpus`, so 2-GPU training must be launched with `torchrun`.

The smoke script handles this automatically:

```bash
uv run torchrun --nproc_per_node=2 ...
```

If `ROBOVLA_NUM_GPUS=1`, it falls back to plain `uv run python`.

## Full Training

Use the full-training wrapper:

```bash
./scripts/train_robovla_robomme.sh
```

By default this uses:

- `HF_HOME=/data1/wlaud1001/huggingface`
- dataset id `Yinpei/robomme_data_lerobot`, resolved from the local `HF_HOME` cache
- `CUDA_VISIBLE_DEVICES=0,1`
- `ROBOVLA_NUM_GPUS=2`
- `ROBOVLA_GLOBAL_BATCH_SIZE=2`
- `ROBOVLA_TRAIN_MODE=action_head`
- `--max-steps 10000`
- `--save-steps 1000`
- output directory `runs/robovla/robomme_gr00t_n1d7_full`

Recommended explicit command:

```bash
HF_HOME=/data1/wlaud1001/huggingface \
CUDA_VISIBLE_DEVICES=0,1 \
ROBOVLA_NUM_GPUS=2 \
ROBOVLA_GLOBAL_BATCH_SIZE=2 \
./scripts/train_robovla_robomme.sh
```

Override training length or output path with normal GR00T finetune flags:

```bash
HF_HOME=/data1/wlaud1001/huggingface \
CUDA_VISIBLE_DEVICES=0,1 \
./scripts/train_robovla_robomme.sh \
  --max-steps 20000 \
  --save-steps 1000 \
  --output-dir runs/robovla/robomme_gr00t_n1d7_20k
```

If you need a specific dataset path, you can still override it:

```bash
ROBOMME_DATASET=/path/to/robomme_data_lerobot_snapshot \
./scripts/train_robovla_robomme.sh
```

## Fine-Tuning Mode

Choose the GR00T fine-tuning mode with `ROBOVLA_TRAIN_MODE`:

- `action_head`: default. Freezes LLM and vision backbone, trains projector and diffusion action head.
- `lora`: freezes base LLM/vision weights, adds LoRA adapters to the GR00T backbone, and also trains projector/action head.
- `full`: full fine-tuning of LLM, vision backbone, projector, and action head. This is the highest-memory option and may OOM on long runs.

Smoke test LoRA first:

```bash
HF_HOME=/data1/wlaud1001/huggingface \
CUDA_VISIBLE_DEVICES=0,1 \
ROBOVLA_NUM_GPUS=2 \
ROBOVLA_GLOBAL_BATCH_SIZE=2 \
ROBOVLA_TRAIN_MODE=lora \
./scripts/train_robovla_robomme_smoke.sh \
  --max-steps 1 \
  --save-steps 1 \
  --output-dir runs/_smoke/gr00t_robomme_lora_1step
```

Full fine-tuning uses the same switch:

```bash
HF_HOME=/data1/wlaud1001/huggingface \
CUDA_VISIBLE_DEVICES=0,1 \
ROBOVLA_TRAIN_MODE=full \
./scripts/train_robovla_robomme.sh \
  --max-steps 1000 \
  --save-steps 500 \
  --output-dir runs/robovla/robomme_gr00t_fullft_1k
```

Or use the full-FT-only wrapper:

```bash
./scripts/train_robovla_robomme_full_ft.sh
```

Common defaults are defined near the top of the script in this form:

```bash
ROBOVLA_GLOBAL_BATCH_SIZE="${ROBOVLA_GLOBAL_BATCH_SIZE:-2}"
ROBOVLA_MAX_STEPS="${ROBOVLA_MAX_STEPS:-10000}"
ROBOVLA_OUTPUT_DIR="${ROBOVLA_OUTPUT_DIR:-$REPO_ROOT/runs/robovla/robomme_gr00t_full_ft}"
```

So you can either edit the default values in the script or override them when
running:

```bash
ROBOVLA_GLOBAL_BATCH_SIZE=1 \
ROBOVLA_MAX_STEPS=1000 \
ROBOVLA_OUTPUT_DIR=runs/robovla/robomme_gr00t_fullft_1k \
./scripts/train_robovla_robomme_full_ft.sh
```

LoRA hyperparameters are environment variables:

- `ROBOVLA_LORA_R`, default `16`
- `ROBOVLA_LORA_ALPHA`, default `32`
- `ROBOVLA_LORA_DROPOUT`, default `0.05`
- `ROBOVLA_LORA_TARGET_MODULES`, default `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## CUDA_HOME Issue

This machine had stale environment variables pointing at CUDA 11.7:

```bash
CUDA_HOME=/usr/local/cuda-11.7
```

But `nvcc` exists at:

```bash
/usr/local/cuda-12.6/bin/nvcc
```

DeepSpeed checks `CUDA_HOME/bin/nvcc`, so the smoke script overrides CUDA paths:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_INC_DIR=$CUDA_HOME/include
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Successful Smoke Result

The following completed successfully:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
ROBOMME_SMOKE_EPISODES=8 \
ROBOVLA_NUM_GPUS=2 \
ROBOVLA_GLOBAL_BATCH_SIZE=2 \
./scripts/train_robovla_robomme_smoke.sh \
  --max-steps 1 \
  --save-steps 1 \
  --output-dir runs/_smoke/gr00t_robomme_2gpu_torchrun_1step
```

Observed:

- GPU 0 and GPU 1 were both used.
- Peak memory was about 27 GB per GPU.
- Training reached 1 step.
- Final log included `train_loss: 1.296875`.
- Output was saved under `runs/_smoke/gr00t_robomme_2gpu_torchrun_1step`.

The following LoRA smoke also completed successfully:

```bash
HF_HOME=/data1/wlaud1001/huggingface \
CUDA_VISIBLE_DEVICES=0,1 \
ROBOVLA_NUM_GPUS=2 \
ROBOVLA_GLOBAL_BATCH_SIZE=2 \
ROBOVLA_TRAIN_MODE=lora \
ROBOMME_SMOKE_EPISODES=2 \
./scripts/train_robovla_robomme_smoke.sh \
  --max-steps 1 \
  --save-steps 1 \
  --output-dir runs/_smoke/gr00t_robomme_lora_1step
```

Observed:

- LoRA adapter was applied to `model.backbone.model`.
- Training reached 1 step.
- Final log included `train_loss: 1.703125`.
- Output was saved under `runs/_smoke/gr00t_robomme_lora_1step`.

## Notes

- Current verified smoke trains the default `action_head` path: projector and
  diffusion action head trainable, LLM and vision backbone frozen.
- LoRA is implemented in the local wrapper by applying PEFT to
  `model.backbone.model`; upstream GR00T files are not modified.
- Full dataset training will spend time generating stats the first time. After
  stats are written, later runs should start faster.
