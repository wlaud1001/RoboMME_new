# Datasets

This repo uses separate dataset paths for raw RoboMME data, generated QwenVL
subgoal data, and GR00T/LeRobot training data. Keep Hugging Face cache files
separate from generated experiment outputs.

## Paths

Use this cache root:

```bash
export HF_HOME=/data1/wlaud1001/huggingface
```

Recommended layout:

```bash
$HF_HOME/hub/                         # Hugging Face-managed cache
$HF_HOME/datasets/robomme_data_h5/    # raw RoboMME H5 files for dataset builds
outputs/                              # generated datasets and experiment artifacts
runs/                                 # training outputs
```

The memory subgoal dataset builder reads raw `.h5` files directly, so it uses:

```bash
/data1/wlaud1001/huggingface/datasets/robomme_data_h5
```

Do not put generated datasets inside `$HF_HOME/hub`; that directory is managed
by Hugging Face tooling.

## QwenVL Subgoal Memory Dataset

The script [scripts/build_robomme_qwenvl_memory_dataset.py](/data3/wlaud1001/workspace/Robotics/RoboMME_new/scripts/build_robomme_qwenvl_memory_dataset.py:1)
builds a separate QwenVL subgoal dataset with sparse visual memory. The official
RoboMME QwenVL builder is left unchanged.

The output dataset is:

```bash
outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/
```

The main grounded training file is:

```bash
outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/grounded_subgoal_train.jsonl
```

### Design

The official QwenVL subgoal predictor uses one view: `front_rgb`. It does not
use `wrist_rgb`. We keep that behavior.

For each training anchor timestep `t`:

- Demo video uses `front_rgb` demo frames, resized to `128x128`.
- Past observations use execution `front_rgb` frames at stride `H=16`, resized
  to `128x128`.
- Current observation uses execution `front_rgb` at timestep `t`, kept at
  `256x256`.
- Past memory keeps at most `K=16` frames.

Past memory indices are:

```text
t - H*K, t - H*(K-1), ..., t - 2H, t - H
```

With defaults:

```text
H = 16
K = 16
past = t-256, t-240, ..., t-32, t-16
current = t
```

Near the beginning of execution, missing past frames are omitted. Demo frames
are not mixed into past observation memory because demo video is provided as a
separate `<video>`.

### Prompt Format

The prompt changes are intentionally minimal. Existing task goal, history, and
question text are preserved. The only added line is `Past observations` when
past frames exist.

Grounded example:

```text
<video>The task goal is: {task_goal}
The history of previous predicted grounded language subgoals are: {history}
Past observations: <image><image>...
<image>What's the next grounded language subgoal based on current observation?
```

The `images` list follows the placeholder order:

```python
images = past_image_paths + [current_image_path]
```

The final `<image>` is always the current observation.

### Pixel Budget

The generated files are mixed resolution:

- past observations: `128x128`
- demo video frames: `128x128`
- current observation: `256x256`

To get the intended token savings during QwenVL training, lower the Qwen/ms-swift
image pixel budget to `128x128` level. Otherwise, the processor may upsample
`128x128` images to `256x256` token grids.

Measured with `Qwen3VLProcessor`:

```text
default shortest_edge=65536:
128 image -> grid [1,16,16] -> 64 visual tokens
256 image -> grid [1,16,16] -> 64 visual tokens

shortest_edge=16384:
128 image -> grid [1,8,8]  -> 16 visual tokens
256 image -> grid [1,16,16] -> 64 visual tokens
```

The desired behavior is the second case.

## Build

Full build:

```bash
HF_HOME=/data1/wlaud1001/huggingface \
/home/wlaud1001/miniconda3/envs/3D/bin/python scripts/build_robomme_qwenvl_memory_dataset.py \
  --raw-data-path /data1/wlaud1001/huggingface/datasets/robomme_data_h5 \
  --preprocessed-data-path outputs/robomme_qwenvl_memory_h16_k16
```

Optional smoke build for one task:

```bash
/home/wlaud1001/miniconda3/envs/3D/bin/python scripts/build_robomme_qwenvl_memory_dataset.py \
  --raw-data-path /data1/wlaud1001/huggingface/datasets/robomme_data_h5 \
  --preprocessed-data-path outputs/_smoke/robomme_qwenvl_memory_h16_k16 \
  --env-filter PickHighlight \
  --max-episodes 1
```

Useful flags:

```bash
--memory-stride 16
--memory-size 16
--past-resolution 128
--current-resolution 256
--demo-resolution 128
--env-filter PickHighlight InsertPeg VideoUnmask
--max-episodes 1
```

## Current Build Result

The full dataset was built successfully at:

```bash
/data3/wlaud1001/workspace/Robotics/RoboMME_new/outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16
```

Build summary:

```text
simple_subgoal_train.jsonl    41997 rows
grounded_subgoal_train.jsonl  41997 rows
image files                   215117
demo videos                   900
total output size             8.0G
```

Validation checks passed:

- JSON rows are `system`, `user`, `assistant`.
- `<image>` count matches `images`.
- `<video>` count matches `videos`.
- Past images are `128x128`.
- Current images are `256x256`.
- Demo videos are `128x128`.
- Maximum image count per row is `17`: `K=16` past observations plus one current
  observation.
- Past observation timesteps are spaced by `H=16`.

## Raw Data Download

If the raw H5 directory is missing, download the dataset into a direct local
directory rather than into `$HF_HOME/hub`:

```bash
export HF_HOME=/data1/wlaud1001/huggingface

huggingface-cli download Yinpei/robomme_data_h5 \
  --repo-type dataset \
  --local-dir $HF_HOME/datasets/robomme_data_h5
```

Then pass:

```bash
--raw-data-path $HF_HOME/datasets/robomme_data_h5
```

Using the Hugging Face hub snapshot path also works if it directly contains the
`.h5` files, but the direct `$HF_HOME/datasets/robomme_data_h5` path is simpler
for scripts that scan files with `os.listdir`.
