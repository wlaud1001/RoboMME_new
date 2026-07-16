from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    gr00t_root = repo_root / "external_dependencies" / "Isaac-GR00T"
    for path in [repo_root / "src", gr00t_root]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = _bootstrap_paths()
DEFAULT_MODALITY_CONFIG_PATH = REPO_ROOT / "src" / "robovla" / "configs" / "robomme_modality.py"

import tyro  # noqa: E402

from gr00t.configs.base_config import get_default_config  # noqa: E402
from gr00t.configs.finetune_config import FinetuneConfig  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

from robovla.robomme_lerobot import (  # noqa: E402
    install_robomme_dataset_adapter,
    prepare_robomme_lerobot_dataset,
)

LOGGER = logging.getLogger(__name__)


def load_modality_config(modality_config_path: str | Path) -> None:
    path = Path(modality_config_path).expanduser().resolve()
    if not path.exists() or path.suffix != ".py":
        raise FileNotFoundError(f"Modality config path does not exist: {path}")

    if str(path.parent) not in sys.path:
        sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    print(f"Loaded modality config: {path}")


def build_gr00t_finetune_config(ft_config: FinetuneConfig):
    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": ft_config.embodiment_tag.value,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    config.model.extra_augmentation_config = (
        json.loads(ft_config.extra_augmentation_config)
        if ft_config.extra_augmentation_config
        else None
    )

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = False

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading
    return config


def _parse_lora_target_modules(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def install_lora_model_adapter() -> None:
    from peft import LoraConfig, get_peft_model
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    original_create_model = Gr00tN1d7Pipeline._create_model
    if getattr(original_create_model, "_robovla_lora_wrapped", False):
        return

    def create_model_with_lora(self):
        model = original_create_model(self)
        lora_config = LoraConfig(
            r=int(os.environ.get("ROBOVLA_LORA_R", "16")),
            lora_alpha=int(os.environ.get("ROBOVLA_LORA_ALPHA", "32")),
            lora_dropout=float(os.environ.get("ROBOVLA_LORA_DROPOUT", "0.05")),
            bias="none",
            target_modules=_parse_lora_target_modules(
                os.environ.get(
                    "ROBOVLA_LORA_TARGET_MODULES",
                    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                )
            ),
        )
        model.backbone.model = get_peft_model(model.backbone.model, lora_config)
        # Keep the language backbone in train mode so LoRA dropout/adapter layers train normally.
        # Base backbone weights remain frozen because PEFT only marks adapter weights trainable.
        model.backbone.tune_llm = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        LOGGER.info(
            "Applied LoRA to GR00T backbone. Trainable parameters: %s / %s (%.2f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            100 * trainable_params / total_params,
        )
        return model

    create_model_with_lora._robovla_lora_wrapped = True
    Gr00tN1d7Pipeline._create_model = create_model_with_lora


def configure_train_mode(ft_config: FinetuneConfig) -> str:
    mode = os.environ.get("ROBOVLA_TRAIN_MODE", "action_head").strip().lower()
    valid_modes = {"action_head", "lora", "full"}
    if mode not in valid_modes:
        raise ValueError(f"ROBOVLA_TRAIN_MODE must be one of {sorted(valid_modes)}, got {mode!r}")

    if mode == "action_head":
        ft_config.tune_llm = False
        ft_config.tune_visual = False
        ft_config.tune_projector = True
        ft_config.tune_diffusion_model = True
    elif mode == "lora":
        ft_config.tune_llm = False
        ft_config.tune_visual = False
        ft_config.tune_projector = True
        ft_config.tune_diffusion_model = True
        install_lora_model_adapter()
    elif mode == "full":
        ft_config.tune_llm = True
        ft_config.tune_visual = True
        ft_config.tune_projector = True
        ft_config.tune_diffusion_model = True

    print(
        "Using ROBOVLA_TRAIN_MODE="
        f"{mode} "
        f"(tune_llm={ft_config.tune_llm}, "
        f"tune_visual={ft_config.tune_visual}, "
        f"tune_projector={ft_config.tune_projector}, "
        f"tune_diffusion_model={ft_config.tune_diffusion_model})"
    )
    return mode


def main(ft_config: FinetuneConfig) -> None:
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/data1/wlaud1001/huggingface"

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    if ft_config.modality_config_path is None:
        ft_config.modality_config_path = str(DEFAULT_MODALITY_CONFIG_PATH)

    configure_train_mode(ft_config)

    for dataset_path in [path for path in ft_config.dataset_path.split(os.pathsep) if path]:
        dataset_info = prepare_robomme_lerobot_dataset(dataset_path)
        print(
            "Prepared RoboMME dataset: "
            f"{dataset_info.root} "
            f"({dataset_info.total_episodes} episodes, {dataset_info.total_frames} frames)"
        )

    load_modality_config(ft_config.modality_config_path)
    install_robomme_dataset_adapter()

    from gr00t.experiment.experiment import run

    run(build_gr00t_finetune_config(ft_config))


if __name__ == "__main__":
    main(tyro.cli(FinetuneConfig, description=__doc__))
