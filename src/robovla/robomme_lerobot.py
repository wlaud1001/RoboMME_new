from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from gr00t.data.dataset.lerobot_episode_loader import LANG_KEYS, LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import ShardedDataset
from gr00t.data.types import ModalityConfig


DEFAULT_DATASET_PATH = Path(
    "/data1/wlaud1001/huggingface/hub/"
    "datasets--Yinpei--robomme_data_lerobot/"
    "snapshots/1510653cccb4d9e5165fb3141c06d88053decc20"
)


@dataclass(frozen=True)
class RoboMMEDatasetInfo:
    root: Path
    modality_path: Path
    total_episodes: int
    total_frames: int


def _default_modality_meta() -> dict[str, Any]:
    return {
        "video": {
            "image": {"original_key": "image"},
            "wrist_image": {"original_key": "wrist_image"},
        },
        "state": {
            "joint_position": {"start": 0, "end": 7, "original_key": "state"},
            "gripper": {"start": 7, "end": 8, "original_key": "state"},
        },
        "action": {
            "joint_position": {"start": 0, "end": 7, "original_key": "actions"},
            "gripper": {"start": 7, "end": 8, "original_key": "actions"},
        },
    }


def prepare_robomme_lerobot_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    overwrite_modality: bool = False,
) -> RoboMMEDatasetInfo:
    """Validate Yinpei/robomme_data_lerobot and add the GR00T modality file."""
    root = Path(dataset_path).expanduser().resolve()
    meta_dir = root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    modality_path = meta_dir / "modality.json"

    for path in [root, meta_dir, info_path, episodes_path, tasks_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required RoboMME dataset path is missing: {path}")

    with info_path.open("r") as f:
        info = json.load(f)

    features = info.get("features", {})
    required_features = ["image", "wrist_image", "state", "actions"]
    missing = [key for key in required_features if key not in features]
    if missing:
        raise ValueError(f"RoboMME dataset is missing required features: {missing}")

    if not modality_path.exists() or overwrite_modality:
        with modality_path.open("w") as f:
            json.dump(_default_modality_meta(), f, indent=2)
            f.write("\n")

    return RoboMMEDatasetInfo(
        root=root,
        modality_path=modality_path,
        total_episodes=int(info.get("total_episodes", 0)),
        total_frames=int(info.get("total_frames", 0)),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_robomme_lerobot_subset(
    source_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path = "runs/_smoke/robomme_lerobot_8eps",
    num_episodes: int = 8,
    overwrite: bool = False,
) -> RoboMMEDatasetInfo:
    """Create a small symlinked RoboMME dataset for fast smoke tests."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()

    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    if output.exists():
        if not overwrite:
            return prepare_robomme_lerobot_dataset(output)
        shutil.rmtree(output)

    prepare_robomme_lerobot_dataset(source)
    source_meta = source / "meta"
    output_meta = output / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    episodes = _read_jsonl(source_meta / "episodes.jsonl")[:num_episodes]
    if len(episodes) < num_episodes:
        raise ValueError(f"Source has only {len(episodes)} episodes, requested {num_episodes}")

    shutil.copy2(source_meta / "tasks.jsonl", output_meta / "tasks.jsonl")
    _write_jsonl(output_meta / "episodes.jsonl", episodes)

    episodes_stats_path = source_meta / "episodes_stats.jsonl"
    if episodes_stats_path.exists():
        _write_jsonl(output_meta / "episodes_stats.jsonl", _read_jsonl(episodes_stats_path)[:num_episodes])

    with (source_meta / "info.json").open("r") as f:
        info = json.load(f)
    info["total_episodes"] = num_episodes
    info["total_frames"] = int(sum(ep["length"] for ep in episodes))
    info["total_chunks"] = 1 + (num_episodes - 1) // int(info["chunks_size"])
    info["splits"] = {"train": f"0:{num_episodes}"}
    with (output_meta / "info.json").open("w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        chunk_idx = episode_index // int(info["chunks_size"])
        rel_path = Path(
            info["data_path"].format(
                episode_chunk=chunk_idx,
                episode_index=episode_index,
            )
        )
        target = output / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source / rel_path)

    return prepare_robomme_lerobot_dataset(output)


def _image_cell_to_rgb(cell: Any) -> Image.Image:
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")

    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return Image.open(BytesIO(cell["bytes"])).convert("RGB")
        if cell.get("path"):
            return Image.open(cell["path"]).convert("RGB")

    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(cell))).convert("RGB")

    raise TypeError(f"Unsupported image cell type: {type(cell)!r}")


class RoboMMELeRobotEpisodeLoader(LeRobotEpisodeLoader):
    """Episode loader for RoboMME's LeRobot parquet files with inline images."""

    def _load_raw_parquet_data(self, episode_index: int) -> pd.DataFrame:
        chunk_idx = episode_index // self.chunk_size
        parquet_filename = self.data_path_pattern.format(
            episode_chunk=chunk_idx,
            episode_index=episode_index,
        )
        return pd.read_parquet(self.dataset_path / parquet_filename)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        df = self._load_parquet_data(episode_id)

        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                df[f"language.{lang_key}"] = self.create_language_from_meta(
                    episode_meta,
                    len(df),
                    lang_key,
                )

        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length].copy()

        if "video" not in self.modality_configs:
            return df

        raw_df = self._load_raw_parquet_data(episode_id).iloc[:actual_length]
        for image_key in self.modality_configs["video"].modality_keys:
            meta_key = self._video_key_mapping.get(image_key, image_key)
            original_key = self.modality_meta["video"][meta_key].get("original_key", meta_key)
            if original_key not in raw_df:
                raise KeyError(
                    f"Image key {original_key!r} not found in episode parquet. "
                    f"Available columns: {list(raw_df.columns)}"
                )
            df[f"video.{image_key}"] = raw_df[original_key].map(_image_cell_to_rgb).tolist()

        return df


class RoboMMEShardedSingleStepDataset(ShardedDataset):
    """GR00T sharded dataset that uses RoboMMELeRobotEpisodeLoader."""

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        shard_size: int = 2**10,
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
    ):
        super().__init__(dataset_path)
        self.embodiment_tag = embodiment_tag
        self.modality_configs = modality_configs
        self.shard_size = shard_size
        self.episode_sampling_rate = episode_sampling_rate
        self.seed = seed
        self.allow_padding = allow_padding
        self.processor = None
        self.rng = np.random.default_rng(seed)
        action_delta_indices = modality_configs["action"].delta_indices
        self.action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1

        self.episode_loader = RoboMMELeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
        )
        self.shard_dataset()

    get_datapoint = ShardedSingleStepDataset.get_datapoint
    get_effective_episode_length = ShardedSingleStepDataset.get_effective_episode_length
    get_initial_actions = ShardedSingleStepDataset.get_initial_actions
    get_shard = ShardedSingleStepDataset.get_shard
    get_shard_length = ShardedSingleStepDataset.get_shard_length
    get_dataset_statistics = ShardedSingleStepDataset.get_dataset_statistics
    shard_dataset = ShardedSingleStepDataset.shard_dataset
    __len__ = ShardedSingleStepDataset.__len__


def install_robomme_dataset_adapter() -> None:
    import gr00t.data.dataset.factory as factory

    factory.ShardedSingleStepDataset = RoboMMEShardedSingleStepDataset
