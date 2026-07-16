from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "src",
        repo_root / "external_dependencies" / "Isaac-GR00T",
        repo_root / "official_baselines" / "robomme_policy_learning" / "examples" / "robomme",
        repo_root / "official_baselines" / "robomme_policy_learning" / "src",
        repo_root / "external_dependencies" / "robomme_benchmark" / "src",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = _bootstrap_paths()

from env_runner import EnvRunner  # noqa: E402
from robovla.robomme_gr00t_eval import RoboMMEGr00tPolicy  # noqa: E402
from utils import RolloutRecorder, TASK_WITH_VIDEO_DEMO  # noqa: E402


def _set_wrapper_default_attr(env, name: str, value) -> None:
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if not hasattr(current, name):
            setattr(current, name, value)
        current = getattr(current, "env", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RoboMME episode with a GR00T policy.")
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "runs_b200/robovla/robomme_gr00t_full_ft/checkpoint-10000"),
    )
    parser.add_argument("--task", default="BinFill")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs_b200/_smoke_eval/gr00t_one_episode"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = RoboMMEGr00tPolicy(model_path=args.model_path, device=args.device)
    env_runner = EnvRunner(args.task, output_dir, max_steps=args.max_steps)
    action_plan: deque[np.ndarray] = deque()

    try:
        env_runner.make_env(args.episode_id)
        # Some RoboMME wrapper revisions read this flag during reset before setting it.
        # Keep the workaround local to this smoke script; do not patch benchmark code.
        _set_wrapper_default_attr(env_runner.env, "current_task_demonstration", False)
        if hasattr(env_runner.env.unwrapped, "agent"):
            _set_wrapper_default_attr(env_runner.env, "agent", env_runner.env.unwrapped.agent)
        initial = env_runner.get_init_obs()
        image = initial["images"][-1]
        wrist_image = initial["wrist_images"][-1]
        state = initial["states"][-1]
        task_goal = initial["task_goal"]
        print(f"task={args.task} episode={args.episode_id} task_goal={task_goal!r}")
        recorder = RolloutRecorder(output_dir / "videos", task_goal, fps=30)
        for idx, (init_image, init_wrist_image, init_state) in enumerate(
            zip(initial["images"], initial["wrist_images"], initial["states"])
        ):
            recorder.record(
                image=init_image.copy(),
                wrist_image=init_wrist_image.copy(),
                state=init_state.copy(),
                action=None,
                is_video_demo=args.task in TASK_WITH_VIDEO_DEMO and idx < len(initial["images"]) - 1,
            )

        outcome = "unknown"
        steps_executed = 0
        for step in range(args.max_steps):
            if not action_plan:
                result = policy.infer(
                    {
                        "observation/image": image,
                        "observation/wrist_image": wrist_image,
                        "observation/state": state,
                        "prompt": task_goal,
                    }
                )
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[-1] != 8:
                    raise ValueError(f"Expected action chunk shape (T, 8), got {actions.shape}")
                print(
                    f"step={step} action_chunk_shape={actions.shape} "
                    f"joint_range=({actions[:, :7].min():.4f}, {actions[:, :7].max():.4f}) "
                    f"gripper_range=({actions[:, 7].min():.4f}, {actions[:, 7].max():.4f})"
                )
                action_plan.extend(actions)

            action = action_plan.popleft()
            obs, stop, outcome = env_runner.step(action)
            steps_executed = step + 1
            if stop:
                print(f"stopped step={step + 1} outcome={outcome}")
                break
            image, wrist_image, state = obs
            recorder.record(
                image=image.copy(),
                wrist_image=wrist_image.copy(),
                state=state.copy(),
                action=action.copy(),
            )
        else:
            print(f"reached max_steps={args.max_steps} outcome={outcome}")

        final_outcome = outcome if outcome != "ongoing" else "timeout"
        video_name = f"{args.task}_ep{args.episode_id}_{final_outcome}.mp4"
        video_path = output_dir / "videos" / video_name
        imageio.mimsave(video_path, recorder.total_images, fps=30, format="FFMPEG")
        result = {
            "task": args.task,
            "episode_id": args.episode_id,
            "model_path": args.model_path,
            "max_steps": args.max_steps,
            "steps_executed": steps_executed,
            "outcome": outcome,
            "video": str(video_path),
        }
        result_path = output_dir / f"{args.task}_ep{args.episode_id}_result.json"
        with result_path.open("w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"saved_video={result['video']}")
        print(f"saved_result={result_path}")
    finally:
        env_runner.close_env()


if __name__ == "__main__":
    main()
