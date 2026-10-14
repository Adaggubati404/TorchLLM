"""YAML-backed model and training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 1024
    vocab_size: int = 50304
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


@dataclass
class TrainConfig:
    out_dir: str = "checkpoints/run"
    batch_size: int = 8
    grad_accum_steps: int = 8
    max_iters: int = 100_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 2_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    save_interval: int = 2_000
    dtype: str = "bfloat16"
    seed: int = 1337
    compile: bool = False
    num_workers: int = 2
    keep_last_k: int = 3
    # Tensor debug logging (shape / dtype / device). Off by default.
    tensor_log: bool = False
    tensor_log_every: int = 1
    tensor_log_max_steps: int = 5  # 0 = unlimited
    tensor_log_model: bool = True
    tensor_log_data: bool = True
    tensor_log_values: bool = False
    tensor_log_full: bool = False
    tensor_log_full_max: int = 32
    tensor_log_blocks: str = "first"  # "first" | "all"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"model": asdict(self.model), "train": asdict(self.train)}


def _filter_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    model = ModelConfig(**_filter_kwargs(ModelConfig, raw.get("model") or {}))
    train = TrainConfig(**_filter_kwargs(TrainConfig, raw.get("train") or {}))
    return Config(model=model, train=train)


def config_from_dict(data: dict[str, Any]) -> Config:
    model = ModelConfig(**_filter_kwargs(ModelConfig, data.get("model") or {}))
    train = TrainConfig(**_filter_kwargs(TrainConfig, data.get("train") or {}))
    return Config(model=model, train=train)
