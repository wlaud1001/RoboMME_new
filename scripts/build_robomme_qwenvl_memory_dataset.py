#!/usr/bin/env python3
"""Build RoboMME QwenVL subgoal data with sparse observation memory.

This keeps the official QwenVL dataset builder intact and writes a separate
dataset directory. Defaults match the current memory design:

- H=16: one past observation every 16 environment steps.
- K=16: at most 16 past observations per sample.
- demo video and past observations are stored at 128x128.
- current observation is stored at 256x256.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import h5py
import imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_SRC = REPO_ROOT / "official_baselines" / "robomme_policy_learning" / "src"
sys.path.insert(0, str(POLICY_SRC))

from mme_vla_suite.dataset_builder.build_vlm_subgoal_dataset_qwenvl import (  # noqa: E402
    GROUNDED_SUBGOAL_SYSTEM_PROMPT,
    SIMPLE_SUBGOAL_SYSTEM_PROMPT,
    DatasetBuilder as QwenVLDatasetBuilder,
)
from mme_vla_suite.dataset_builder.robomme_h5_utils import (  # noqa: E402
    get_env_id_from_filename,
    get_episode_indices,
    get_task_goal,
    get_timestep_indices,
)


class MemoryQwenVLDatasetBuilder(QwenVLDatasetBuilder):
    def __init__(
        self,
        *args,
        memory_stride: int = 16,
        memory_size: int = 16,
        past_resolution: int = 128,
        current_resolution: int = 256,
        demo_resolution: int = 128,
        env_filter: set[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.memory_stride = memory_stride
        self.memory_size = memory_size
        self.past_resolution = past_resolution
        self.current_resolution = current_resolution
        self.demo_resolution = demo_resolution
        self.env_filter = env_filter
        self._past_image_cache: dict[tuple[str, int, int], str] = {}

    def run(self) -> list:
        results: list = []
        for file in os.listdir(self.raw_data_path):
            if not file.endswith(".h5"):
                continue
            env_id = get_env_id_from_filename(file)
            if self.env_filter and env_id not in self.env_filter:
                continue

            print(f"\nprocessing file: {file}")
            with h5py.File(os.path.join(self.raw_data_path, file), "r") as data:
                episode_indices = get_episode_indices(data, self.max_episodes)
                for episode_idx in episode_indices:
                    results.append(self.process_per_episode(data, env_id, episode_idx))
        return results

    @staticmethod
    def _resize_square(image: np.ndarray, resolution: int) -> np.ndarray:
        if image.shape[0] == resolution and image.shape[1] == resolution:
            return image
        return cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)

    def _write_image(self, image: np.ndarray, path: str, resolution: int) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(path, self._resize_square(image, resolution))
        return path

    def _write_demo_video(
        self,
        episode_data: h5py.Group,
        env_id: str,
        episode_idx: int,
        exec_start_idx: int,
    ) -> str | None:
        if exec_start_idx <= 0:
            return None

        video_frames = [
            self._resize_square(
                episode_data[f"timestep_{i}"]["obs"]["front_rgb"][()],
                self.demo_resolution,
            )
            for i in range(exec_start_idx)
        ]
        video_path = os.path.join(self.images_dir, f"{env_id}_ep{episode_idx}_demo128.mp4")
        imageio.mimsave(video_path, video_frames, fps=30)
        return video_path

    def _past_indices(self, idx: int, exec_start_idx: int) -> list[int]:
        candidates = [
            idx - self.memory_stride * offset
            for offset in range(self.memory_size, 0, -1)
        ]
        return [past_idx for past_idx in candidates if past_idx >= exec_start_idx]

    def _past_image_paths(
        self,
        episode_data: h5py.Group,
        env_id: str,
        episode_idx: int,
        idx: int,
        exec_start_idx: int,
    ) -> list[str]:
        paths: list[str] = []
        for past_idx in self._past_indices(idx, exec_start_idx):
            key = (env_id, episode_idx, past_idx)
            cached = self._past_image_cache.get(key)
            if cached is None:
                image = episode_data[f"timestep_{past_idx}"]["obs"]["front_rgb"][()]
                cached = os.path.join(
                    self.images_dir,
                    f"{env_id}_ep{episode_idx}_past128_step{past_idx}.png",
                )
                self._write_image(image, cached, self.past_resolution)
                self._past_image_cache[key] = cached
            paths.append(cached)
        return paths

    def _current_image_path(
        self,
        episode_data: h5py.Group,
        env_id: str,
        episode_idx: int,
        idx: int,
    ) -> str:
        image = episode_data[f"timestep_{idx}"]["obs"]["front_rgb"][()]
        image_path = os.path.join(
            self.images_dir, f"{env_id}_ep{episode_idx}_current256_step{idx}.png"
        )
        return self._write_image(image, image_path, self.current_resolution)

    @staticmethod
    def _image_placeholders(count: int) -> str:
        return "".join(["<image>" for _ in range(count)])

    def make_simple_subgoal_data(
        self,
        task_goal: str,
        subgoal: str,
        current_image_path: str,
        video_path: str | None = None,
        past_image_paths: list[str] | None = None,
    ) -> dict:
        past_image_paths = past_image_paths or []
        video_prefix = "<video>" if video_path else ""
        past_prompt = (
            f"Past observations: {self._image_placeholders(len(past_image_paths))}\n"
            if past_image_paths
            else ""
        )

        if len(self.history_simple_subgoals) == 0:
            user_prompt = (
                f"{video_prefix}The task goal is: {task_goal}\n"
                "This is the initial turn for prediction\n"
                f"{past_prompt}"
                "<image>What's the next language subgoal based on current observation?"
            )
        else:
            user_prompt = (
                f"{video_prefix}The task goal is: {task_goal}\n"
                f"The history of previous predicted language subgoals are: {self._wrap_history_subgoals(self.history_simple_subgoals)}\n"
                f"{past_prompt}"
                "<image>What's the next language subgoal based on current observation?"
            )

        result = {
            "messages": [
                {"role": "system", "content": SIMPLE_SUBGOAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": subgoal},
            ],
            "images": past_image_paths + [current_image_path],
            "videos": [video_path] if video_path else [],
        }

        if self.history_simple_subgoals:
            if self.history_simple_subgoals[-1] != subgoal:
                self.history_simple_subgoals.append(subgoal)
        else:
            self.history_simple_subgoals.append(subgoal)

        return result

    def make_grounded_subgoal_data(
        self,
        task_goal: str,
        subgoal: str,
        current_image_path: str,
        video_path: str | None = None,
        past_image_paths: list[str] | None = None,
    ) -> dict:
        past_image_paths = past_image_paths or []
        video_prefix = "<video>" if video_path else ""
        past_prompt = (
            f"Past observations: {self._image_placeholders(len(past_image_paths))}\n"
            if past_image_paths
            else ""
        )
        assistant_prompt, bbox = self._preprocess_grounded_subgoal(subgoal)

        if len(self.history_grounded_subgoals) == 0:
            user_prompt = (
                f"{video_prefix}The task goal is: {task_goal}\n"
                "This is the initial turn for prediction\n"
                f"{past_prompt}"
                "<image>What's the next grounded language subgoal based on current observation?"
            )
        else:
            user_prompt = (
                f"{video_prefix}The task goal is: {task_goal}\n"
                f"The history of previous predicted grounded language subgoals are: {self._wrap_history_subgoals(self.history_grounded_subgoals)}\n"
                f"{past_prompt}"
                "<image>What's the next grounded language subgoal based on current observation?"
            )

        result = {
            "messages": [
                {"role": "system", "content": GROUNDED_SUBGOAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prompt},
            ],
            "objects": {
                "ref": [],
                "bbox": self._add_noise_to_bbox(self.history_grounded_bboxes + bbox),
            },
            "images": past_image_paths + [current_image_path],
            "videos": [video_path] if video_path else [],
        }

        if self.history_grounded_subgoals:
            if self.history_grounded_subgoals[-1] != assistant_prompt:
                self.history_grounded_subgoals.append(assistant_prompt)
                self.history_grounded_bboxes.extend(bbox)
        else:
            self.history_grounded_subgoals.append(assistant_prompt)
            self.history_grounded_bboxes.extend(bbox)

        return result

    def process_per_episode(
        self,
        env_dataset: h5py.File,
        env_id: str,
        episode_idx: int,
    ) -> None:
        print(f"processing episode {episode_idx} of {env_id}...")
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []
        self._past_image_cache = {}

        episode_data = env_dataset[f"episode_{episode_idx}"]
        task_goal = get_task_goal(episode_data, lower=True)
        timestep_indices = get_timestep_indices(episode_data)
        exec_start_idx = self._first_execution_step(episode_data)

        transition_idxs = self._compute_transition_idxs(
            episode_data, env_id, exec_start_idx, timestep_indices
        )
        if transition_idxs[-1] != len(timestep_indices) - 1:
            transition_idxs.append(len(timestep_indices) - 1)
        print("transition_idxs: ", transition_idxs)

        select_idxs, duplicate_idxs = self._compute_select_and_duplicate_idxs(
            transition_idxs, len(timestep_indices), env_id
        )
        print("select_idxs: ", select_idxs)

        video_path = self._write_demo_video(
            episode_data, env_id, episode_idx, exec_start_idx
        )

        last_simple_subgoal = None
        last_grounded_subgoal = None

        for idx in select_idxs:
            simple_subgoal = (
                episode_data[f"timestep_{idx}"]["info"]["simple_subgoal"][()]
                .decode()
                .lower()
            )
            grounded_subgoal = (
                episode_data[f"timestep_{idx}"]["info"]["grounded_subgoal"][()]
                .decode()
                .lower()
            )

            if "complete" in simple_subgoal:
                simple_subgoal = last_simple_subgoal
            if "complete" in grounded_subgoal:
                grounded_subgoal = last_grounded_subgoal

            if simple_subgoal is None or grounded_subgoal is None:
                last_simple_subgoal = simple_subgoal
                last_grounded_subgoal = grounded_subgoal
                continue

            past_image_paths = self._past_image_paths(
                episode_data, env_id, episode_idx, idx, exec_start_idx
            )
            current_image_path = self._current_image_path(
                episode_data, env_id, episode_idx, idx
            )

            simple_subgoal_data = self.make_simple_subgoal_data(
                task_goal,
                simple_subgoal,
                current_image_path,
                video_path,
                past_image_paths,
            )
            grounded_subgoal_data = self.make_grounded_subgoal_data(
                task_goal,
                grounded_subgoal,
                current_image_path,
                video_path,
                past_image_paths,
            )

            self._append_training_rows(simple_subgoal_data, grounded_subgoal_data)

            dup_count = duplicate_idxs.get(idx, 0)
            if dup_count > 0:
                print(f"duplicate {idx} for {dup_count} more times")
                self._append_training_rows(
                    simple_subgoal_data, grounded_subgoal_data, times=dup_count
                )

            last_simple_subgoal = simple_subgoal
            last_grounded_subgoal = grounded_subgoal


def _parse_args() -> argparse.Namespace:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    parser = argparse.ArgumentParser(
        description="Build QwenVL subgoal JSONL with sparse past-observation memory."
    )
    parser.add_argument(
        "--raw-data-path",
        default=os.path.join(hf_home, "datasets", "robomme_data_h5"),
        help="Directory containing RoboMME HDF5 files.",
    )
    parser.add_argument(
        "--preprocessed-data-path",
        default=str(REPO_ROOT / "outputs" / "robomme_qwenvl_memory_h16_k16"),
        help="Output root; dataset is written under its qwenvl_memory_h16_k16 subdir.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--env-filter",
        nargs="*",
        default=None,
        help="Optional env IDs to include, e.g. PickHighlight InsertPeg VideoUnmask.",
    )
    parser.add_argument("--memory-stride", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=16)
    parser.add_argument("--past-resolution", type=int, default=128)
    parser.add_argument("--current-resolution", type=int, default=256)
    parser.add_argument("--demo-resolution", type=int, default=128)
    parser.add_argument(
        "--vlm-dir-name",
        default="qwenvl_memory_h16_k16",
        help="Subdirectory name under preprocessed data path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.perf_counter()
    builder = MemoryQwenVLDatasetBuilder(
        raw_data_path=args.raw_data_path,
        preprocessed_data_path=args.preprocessed_data_path,
        max_episodes=args.max_episodes,
        visualize=False,
        vlm_dir_name=args.vlm_dir_name,
        memory_stride=args.memory_stride,
        memory_size=args.memory_size,
        past_resolution=args.past_resolution,
        current_resolution=args.current_resolution,
        demo_resolution=args.demo_resolution,
        env_filter=set(args.env_filter) if args.env_filter else None,
    )
    builder.run()
    print(f"Time taken: {(time.perf_counter() - t0) / 60:.2f} minutes")
    print(f"Simple JSONL: {builder.simple_subgoal_train_data_path}")
    print(f"Grounded JSONL: {builder.grounded_subgoal_train_data_path}")


if __name__ == "__main__":
    main()
