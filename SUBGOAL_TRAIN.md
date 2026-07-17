# Subgoal Predictor Training

This repo uses separate dataset paths for raw RoboMME data, generated QwenVL
subgoal data, and GR00T/LeRobot training data. Keep Hugging Face cache files
separate from generated experiment outputs.

## Command Flow

아래 순서대로 실행하면 됩니다. 경로는 모두 `$HF_HOME` 기준입니다.

```bash
# 0. Set paths.
cd /path/to/RoboMME_new
export HF_HOME=/path/to/huggingface
export PYTHON_BIN=/path/to/python
```

```bash
# 1. Download raw RoboMME H5 dataset, decompress .h5.tar.xz to .h5,
#    download official QwenVL grounded subgoal adapter zip, and unzip it.
./scripts/prepare_robomme_qwenvl_assets.sh
```

```bash
# 2. Smoke-build one task first.
$PYTHON_BIN scripts/build_robomme_qwenvl_memory_dataset.py \
  --raw-data-path "$HF_HOME/datasets/robomme_data_h5" \
  --preprocessed-data-path outputs/_smoke/robomme_qwenvl_memory_h16_k16 \
  --env-filter PickHighlight \
  --max-episodes 1
```

```bash
# 3. Build the full sparse-memory QwenVL subgoal dataset.
#    This keeps the K=16 version as the base dataset.
$PYTHON_BIN scripts/build_robomme_qwenvl_memory_dataset.py \
  --raw-data-path "$HF_HOME/datasets/robomme_data_h5" \
  --preprocessed-data-path outputs/robomme_qwenvl_memory_h16_k16
```

```bash
# 4. Optional: materialize a smaller-K JSONL without rebuilding images/videos.
#    For training, always materialize once even for K=16 so every row has the
#    same swift/Hugging Face dataset schema.
$PYTHON_BIN scripts/materialize_qwenvl_memory_k.py \
  --input-jsonl outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/grounded_subgoal_train.jsonl \
  --output-jsonl outputs/robomme_qwenvl_memory_h16_k16_swift/grounded_subgoal_train.jsonl \
  --memory-size 16
```

```bash
# 5. Train the grounded subgoal predictor.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/train_qwenvl_subgoal_memory.sh
```

For a short training smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 \
QWENVL_PER_DEVICE_BATCH_SIZE=1 \
QWENVL_GRADIENT_ACCUMULATION_STEPS=1 \
QWENVL_MAX_STEPS=5 \
QWENVL_SAVE_STEPS=5 \
QWENVL_OUTPUT_DIR=runs/_smoke/qwenvl_subgoal_memory \
./scripts/train_qwenvl_subgoal_memory.sh
```

For a configurable full run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
QWENVL_PER_DEVICE_BATCH_SIZE=2 \
QWENVL_GRADIENT_ACCUMULATION_STEPS=8 \
QWENVL_MAX_STEPS=1000 \
QWENVL_SAVE_STEPS=100 \
QWENVL_OUTPUT_DIR=runs/qwenvl_subgoal_memory_h16_k16/grounded_1k \
./scripts/train_qwenvl_subgoal_memory.sh
```

## Dataset Prepare Section

### Paths

Use this cache root:

```bash
export HF_HOME=/path/to/huggingface
```

Recommended layout:

```bash
$HF_HOME/hub/                         # Hugging Face-managed cache
$HF_HOME/datasets/robomme_data_h5/    # raw RoboMME H5 files for dataset builds
$HF_HOME/downloads/vlm_subgoal_predictor/
                                      # downloaded official adapter zip files
$HF_HOME/models/robomme/              # extracted RoboMME model/adapters
outputs/                              # generated datasets and experiment artifacts
runs/                                 # training outputs
```

The memory subgoal dataset builder reads raw `.h5` files directly, so it uses:

```bash
$HF_HOME/datasets/robomme_data_h5
```

Do not put generated datasets inside `$HF_HOME/hub`; that directory is managed
by Hugging Face tooling.

### Prepare Official Assets

Use `scripts/prepare_robomme_qwenvl_assets.sh` to place both the raw H5 dataset
and the official QwenVL grounded subgoal adapter under `$HF_HOME`.

```bash
export HF_HOME=/path/to/huggingface

./scripts/prepare_robomme_qwenvl_assets.sh
```

Default prepared paths:

```bash
$HF_HOME/datasets/robomme_data_h5/
$HF_HOME/datasets/robomme_data_h5/tarxz_h5.py
$HF_HOME/downloads/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200.zip
$HF_HOME/models/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200/
```

The raw dataset repository may contain per-file `.h5.tar.xz` archives. The
prepare script automatically runs:

```bash
python "$HF_HOME/datasets/robomme_data_h5/tarxz_h5.py" decompress \
  --input_dir "$HF_HOME/datasets/robomme_data_h5" \
  --jobs 16
```

If no `.h5.tar.xz` archives are present, this step is skipped. Set
`ROBOMME_DECOMPRESS_RAW_DATA=false` to disable raw data decompression, or set
`ROBOMME_REMOVE_RAW_ARCHIVE=true` to remove archives after successful extraction.

The zip file is downloaded to `$HF_HOME/downloads/vlm_subgoal_predictor`.
Because the archive contains a top-level `checkpoint-1200/` directory, it is
extracted into:

```bash
$HF_HOME/models/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/
```

So the final adapter path used by training is:

```bash
$HF_HOME/models/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200
```

Useful variants:

```bash
./scripts/prepare_robomme_qwenvl_assets.sh --skip-raw-data
./scripts/prepare_robomme_qwenvl_assets.sh --skip-model
./scripts/prepare_robomme_qwenvl_assets.sh --force-unzip
```

Override paths with environment variables if the storage layout changes:

```bash
ROBOMME_RAW_DATA_DIR=/path/to/robomme_data_h5 \
ROBOMME_MODEL_ROOT=/path/to/models/robomme \
ROBOMME_DECOMPRESS_JOBS=32 \
./scripts/prepare_robomme_qwenvl_assets.sh
```

## QwenVL Subgoal Memory Dataset

The script `scripts/build_robomme_qwenvl_memory_dataset.py`
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
export HF_HOME=/path/to/huggingface
PYTHON_BIN=/path/to/python

$PYTHON_BIN scripts/build_robomme_qwenvl_memory_dataset.py \
  --raw-data-path "$HF_HOME/datasets/robomme_data_h5" \
  --preprocessed-data-path outputs/robomme_qwenvl_memory_h16_k16
```

This creates and keeps the base `H=16, K=16` dataset:

```bash
outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/grounded_subgoal_train.jsonl
```

Before training, materialize the Swift-compatible K=16 JSONL:

```bash
$PYTHON_BIN scripts/materialize_qwenvl_memory_k.py \
  --input-jsonl outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/grounded_subgoal_train.jsonl \
  --output-jsonl outputs/robomme_qwenvl_memory_h16_k16_swift/grounded_subgoal_train.jsonl \
  --memory-size 16
```

This keeps the original K=16 dataset unchanged and writes the training JSONL to:

```bash
outputs/robomme_qwenvl_memory_h16_k16_swift/grounded_subgoal_train.jsonl
```

Optional smoke build for one task:

```bash
PYTHON_BIN=/path/to/python

$PYTHON_BIN scripts/build_robomme_qwenvl_memory_dataset.py \
  --raw-data-path "$HF_HOME/datasets/robomme_data_h5" \
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

## Materialize K-Specific JSONL

Use `scripts/materialize_qwenvl_memory_k.py` when you want to change `K`
without rebuilding images, videos, or reading HDF5 files again. This keeps
`H=16` fixed and selects the last `K` past observations from the base K=16
JSONL.

Materialization is also used for `K=16` because Hugging Face `datasets` requires
all JSONL rows to have identical columns. The materialized files always include
`videos`, using an empty list when a row has no demo video.

If another server already ran the older heavy build, do not rebuild unless the
media files are missing. Keep the existing base dataset:

```bash
outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/grounded_subgoal_train.jsonl
outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/images/
```

Then run materialization on that server:

```bash
$PYTHON_BIN scripts/materialize_qwenvl_memory_k.py \
  --input-jsonl outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/grounded_subgoal_train.jsonl \
  --output-jsonl outputs/robomme_qwenvl_memory_h16_k16_swift/grounded_subgoal_train.jsonl \
  --memory-size 16
```

Heavy rebuild is only needed if the base JSONL or `images/` directory is missing,
the media paths point to files that do not exist on that server, or you changed
`H`, max source `K`, resolutions, prompt format, or raw H5 processing.

Example for `K=8`:

```bash
$PYTHON_BIN scripts/materialize_qwenvl_memory_k.py \
  --input-jsonl outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/grounded_subgoal_train.jsonl \
  --output-jsonl outputs/robomme_qwenvl_memory_h16_k8/grounded_subgoal_train.jsonl \
  --memory-size 8
```

Example for `K=4`:

```bash
$PYTHON_BIN scripts/materialize_qwenvl_memory_k.py \
  --input-jsonl outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16/grounded_subgoal_train.jsonl \
  --output-jsonl outputs/robomme_qwenvl_memory_h16_k4/grounded_subgoal_train.jsonl \
  --memory-size 4
```

Then train on the materialized JSONL by overriding `QWENVL_DATASET_PATH`:

```bash
QWENVL_DATASET_PATH=outputs/robomme_qwenvl_memory_h16_k8/grounded_subgoal_train.jsonl \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/train_qwenvl_subgoal_memory.sh
```

The original K=16 dataset remains unchanged. This materialization step only
rewrites JSONL rows:

- `images` becomes `last K past images + current image`.
- The `Past observations: <image>...` prompt line is updated to match K.
- `videos` is always present; rows without video use `videos: []`.
- `images` and `videos` are written as absolute paths so training still works
  after the script changes into the official policy repo directory.
- Assistant labels, system prompts, and grounded bbox fields are kept.
- Early timesteps may have fewer than K past observations because execution has
  just started.

## Current Build Result

The full dataset was built successfully at:

```bash
outputs/robomme_qwenvl_memory_h16_k16/qwenvl_memory_h16_k16
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

## Train Subgoal Predictor

Use `scripts/train_qwenvl_subgoal_memory.sh`
to fine-tune the grounded QwenVL subgoal predictor on this memory dataset.

```bash
HF_HOME=/path/to/huggingface \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/train_qwenvl_subgoal_memory.sh
```

Important variables are defined near the top of the script and can be overridden
from the shell:

```bash
QWENVL_PER_DEVICE_BATCH_SIZE=2 \
QWENVL_GRADIENT_ACCUMULATION_STEPS=8 \
QWENVL_MAX_STEPS=1000 \
QWENVL_SAVE_STEPS=100 \
QWENVL_OUTPUT_DIR=runs/qwenvl_subgoal_memory_h16_k16/grounded_1k \
./scripts/train_qwenvl_subgoal_memory.sh
```

By default, this continues from the official RoboMME grounded QwenVL subgoal
predictor adapter:

```bash
QWENVL_INIT_ADAPTER=$HF_HOME/models/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200
```

If this adapter path does not exist, the training script automatically calls
`scripts/prepare_robomme_qwenvl_assets.sh --skip-raw-data` and extracts the
official adapter under `$HF_HOME/models/robomme`. Set
`QWENVL_AUTO_PREPARE_ADAPTER=false` to disable that behavior.

The script passes this path to `swift sft` with the full `--adapters` option.
Do not shorten it to `--adapter`; recent ms-swift versions treat that as an
ambiguous abbreviation.

Set `QWENVL_INIT_ADAPTER=""` to train from the base `Qwen/Qwen3-VL-4B-Instruct`
model instead.

The script sets the visual token budget for this dataset:

```bash
IMAGE_MAX_TOKEN_NUM=64
VIDEO_MAX_TOKEN_NUM=16
FPS_MAX_FRAMES=10
```

This keeps current observations at `256x256` token resolution while allowing
past observations and demo video frames saved at `128x128` to use fewer tokens.
The script also passes `--use_hf true` to `swift sft`, so the base Qwen model is
resolved through Hugging Face tooling and `$HF_HOME` instead of ModelScope.
Run it from an environment where `swift` is installed, or set:

```bash
SWIFT_BIN=/path/to/swift
```

## Manual Raw Data Download

If the raw H5 directory is missing, download the dataset into a direct local
directory rather than into `$HF_HOME/hub`:

```bash
export HF_HOME=/path/to/huggingface

huggingface-cli download Yinpei/robomme_data_h5 \
  --repo-type dataset \
  --local-dir $HF_HOME/datasets/robomme_data_h5
```

If the dataset contains `.h5.tar.xz` archives, decompress them before building:

```bash
python "$HF_HOME/datasets/robomme_data_h5/tarxz_h5.py" decompress \
  --input_dir "$HF_HOME/datasets/robomme_data_h5" \
  --jobs 16
```

Then pass:

```bash
--raw-data-path $HF_HOME/datasets/robomme_data_h5
```

Using the Hugging Face hub snapshot path also works if it directly contains the
`.h5` files, but the direct `$HF_HOME/datasets/robomme_data_h5` path is simpler
for scripts that scan files with `os.listdir`.

## Final Training Command

After preparing assets and building the dataset, train with:

```bash
cd /path/to/RoboMME_new
export HF_HOME=/path/to/huggingface

CUDA_VISIBLE_DEVICES=0,1,2,3 \
QWENVL_PER_DEVICE_BATCH_SIZE=2 \
QWENVL_GRADIENT_ACCUMULATION_STEPS=8 \
QWENVL_MAX_STEPS=1000 \
QWENVL_SAVE_STEPS=100 \
QWENVL_OUTPUT_DIR=runs/qwenvl_subgoal_memory_h16_k16/grounded_1k \
./scripts/train_qwenvl_subgoal_memory.sh
```

The script defaults to:

```bash
QWENVL_DATASET_PATH=outputs/robomme_qwenvl_memory_h16_k16_swift/grounded_subgoal_train.jsonl
QWENVL_INIT_ADAPTER=$HF_HOME/models/robomme/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200
```

## Notes

- Set `HF_HOME` before running prepare, build, or train. Dataset/model paths are
  derived from `$HF_HOME`; do not hard-code machine-specific paths.
- The build step needs extracted `.h5` files, not only `.h5.tar.xz` archives.
  `scripts/prepare_robomme_qwenvl_assets.sh` handles this automatically.
- The model zip is stored under `$HF_HOME/downloads/...`; the extracted adapter
  used by training is under `$HF_HOME/models/robomme/.../checkpoint-1200`.
- Do not store generated datasets in `$HF_HOME/hub`; use `outputs/` for builds
  and `runs/` for training outputs.
- Run the smoke dataset build before the full build when changing paths or
  environment variables.
- Run `scripts/materialize_qwenvl_memory_k.py --memory-size 16` before training
  the default K=16 setup; this normalizes JSONL columns for Hugging Face
  `datasets`.
- Run the short training smoke command before a long run when using a new GPU
  setup or batch size.
- If `swift` is not on `PATH`, activate the ms-swift environment or set
  `SWIFT_BIN=/path/to/swift`.
- GPU selection should be done with `CUDA_VISIBLE_DEVICES`; do not hard-code GPU
  indices inside scripts.
- If the official adapter is missing, the training script will try to prepare it
  automatically. Set `QWENVL_AUTO_PREPARE_ADAPTER=false` to disable that.
- Use `--adapters` for ms-swift adapter loading. `--adapter` is ambiguous in
  recent ms-swift CLIs.
- The training script defaults to `QWENVL_USE_HF=true`, which passes
  `--use_hf true` to ms-swift and keeps base model resolution under `$HF_HOME`.
