from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_gr00t() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    gr00t_root = repo_root / "external_dependencies" / "Isaac-GR00T"
    for path in [repo_root / "src", gr00t_root]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _last(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[-1]
    return value


def _rgb_uint8(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {array.shape}")
    return array


def _task_text(inputs: dict[str, Any]) -> str:
    value = inputs.get("task_goal", inputs.get("prompt", inputs.get("task", "")))
    if isinstance(value, (list, tuple)):
        value = value[0]
    if not isinstance(value, str):
        raise TypeError(f"Task text must be str, got {type(value)}")
    return value


def _packed_state(inputs: dict[str, Any]) -> np.ndarray:
    if "observation/state" in inputs:
        state = np.asarray(inputs["observation/state"], dtype=np.float32)
    elif "states" in inputs:
        state = np.asarray(_last(inputs["states"]), dtype=np.float32)
    elif "joint_state_list" in inputs and "gripper_state_list" in inputs:
        joint = np.asarray(_last(inputs["joint_state_list"]), dtype=np.float32)
        gripper = np.asarray(_last(inputs["gripper_state_list"]), dtype=np.float32)
        state = np.concatenate([joint[:7], gripper[:1]], axis=0, dtype=np.float32)
    else:
        raise KeyError(
            "Expected one of: 'observation/state', 'states', or "
            "'joint_state_list' + 'gripper_state_list'"
        )
    if state.shape != (8,):
        raise ValueError(f"Packed RoboMME state must have shape (8,), got {state.shape}")
    return state


def robomme_inputs_to_gr00t_observation(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert RoboMME observation dict to the nested format expected by Gr00tPolicy."""
    image = _rgb_uint8(
        _last(inputs["front_rgb_list"]) if "front_rgb_list" in inputs else inputs["observation/image"],
        "front image",
    )
    wrist_image = _rgb_uint8(
        _last(inputs["wrist_rgb_list"])
        if "wrist_rgb_list" in inputs
        else inputs["observation/wrist_image"],
        "wrist image",
    )
    state = _packed_state(inputs)

    return {
        "video": {
            "image": image[None, None],
            "wrist_image": wrist_image[None, None],
        },
        "state": {
            "joint_position": state[:7].reshape(1, 1, 7).astype(np.float32),
            "gripper": state[7:8].reshape(1, 1, 1).astype(np.float32),
        },
        "language": {
            "task": [[_task_text(inputs)]],
        },
    }


def decoded_gr00t_action_to_robomme(gr00t_action: dict[str, np.ndarray]) -> np.ndarray:
    """Convert decoded GR00T action to RoboMME absolute joint_angle action chunk.

    The input must be the output of ``Gr00tPolicy.get_action()``. It is already
    denormalized, and relative joint deltas have already been converted back to
    absolute joint positions by GR00T's saved processor. Do not add the current
    joint state here.
    """
    joint = np.asarray(gr00t_action["joint_position"], dtype=np.float32)
    gripper = np.asarray(gr00t_action["gripper"], dtype=np.float32)
    if joint.ndim == 3:
        joint = joint[0]
    if gripper.ndim == 3:
        gripper = gripper[0]
    if joint.ndim != 2 or joint.shape[-1] != 7:
        raise ValueError(f"Decoded joint action must have shape (T, 7), got {joint.shape}")
    if gripper.ndim != 2 or gripper.shape[-1] != 1:
        raise ValueError(f"Decoded gripper action must have shape (T, 1), got {gripper.shape}")
    if joint.shape[0] != gripper.shape[0]:
        raise ValueError(f"Action horizon mismatch: {joint.shape[0]} vs {gripper.shape[0]}")
    return np.concatenate([joint, gripper], axis=-1).astype(np.float32)


class RoboMMEGr00tPolicy:
    """RoboMME policy wrapper for a finetuned GR00T checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        embodiment_tag: str = "NEW_EMBODIMENT",
    ) -> None:
        _bootstrap_gr00t()
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(model_path),
            device=device,
        )

    def infer(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        gr00t_observation = robomme_inputs_to_gr00t_observation(inputs)
        gr00t_action, _ = self.policy.get_action(gr00t_observation)
        return {"actions": decoded_gr00t_action_to_robomme(gr00t_action)}

    def reset(self) -> dict[str, bool]:
        self.policy.reset()
        return {"reset_finished": True}
