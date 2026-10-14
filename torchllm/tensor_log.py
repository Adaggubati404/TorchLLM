"""Config-controlled tensor debug logging (shape, dtype, device, optional stats)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TensorLogConfig:
    enabled: bool = False
    every: int = 1
    max_steps: int = 5  # 0 = unlimited training steps logged
    model: bool = True
    data: bool = True
    values: bool = False
    full: bool = False
    full_max: int = 32
    blocks: str = "first"  # "first" | "all"

    def __post_init__(self) -> None:
        if self.blocks not in ("first", "all"):
            raise ValueError(f"tensor_log_blocks must be 'first' or 'all', got {self.blocks!r}")
        if self.every < 1:
            raise ValueError(f"tensor_log_every must be >= 1, got {self.every}")


class TensorLogger:
    def __init__(self, cfg: TensorLogConfig | None = None) -> None:
        self.cfg = cfg or TensorLogConfig()
        self._step: int = 0
        self._steps_logged: int = 0
        self._active: bool = False
        self._gen_tokens_logged: int = 0

    def configure(self, cfg: TensorLogConfig) -> None:
        self.cfg = cfg
        self._step = 0
        self._steps_logged = 0
        self._active = False
        self._gen_tokens_logged = 0
        if self.cfg.enabled:
            print(
                f"[tensor] enabled every={self.cfg.every} max_steps={self.cfg.max_steps} "
                f"model={self.cfg.model} data={self.cfg.data} values={self.cfg.values} "
                f"full={self.cfg.full} blocks={self.cfg.blocks}",
                flush=True,
            )

    def begin_step(self, it: int) -> None:
        """Call at the start of each training optimizer step."""
        self._step = it
        if not self.cfg.enabled:
            self._active = False
            return
        if self.cfg.max_steps > 0 and self._steps_logged >= self.cfg.max_steps:
            self._active = False
            return
        if it % self.cfg.every != 0:
            self._active = False
            return
        self._active = True
        self._steps_logged += 1

    def enable_for_run(self) -> None:
        """Activate for a one-shot run (generate) with no step budgeting unless set."""
        if not self.cfg.enabled:
            self._active = False
            return
        self._active = True
        self._gen_tokens_logged = 0

    def should_log_model(self) -> bool:
        return self.cfg.enabled and self._active and self.cfg.model

    def should_log_data(self) -> bool:
        return self.cfg.enabled and self._active and self.cfg.data

    def should_log_block(self, layer_idx: int) -> bool:
        if not self.should_log_model():
            return False
        if self.cfg.blocks == "all":
            return True
        return layer_idx == 0

    def should_log_generate_token(self) -> bool:
        """Throttle generate-time model logs to the first few tokens."""
        if not self.cfg.enabled or not self.cfg.model:
            self._active = False
            return False
        limit = self.cfg.max_steps if self.cfg.max_steps > 0 else 5
        if self._gen_tokens_logged >= limit:
            self._active = False
            return False
        self._gen_tokens_logged += 1
        self._active = True
        return True

    def force_log_tensor(self, name: str, tensor: Any, where: str = "") -> None:
        """Log once even if step budget inactive (still requires master switch)."""
        if not self.cfg.enabled:
            return
        prev = self._active
        self._active = True
        self.log_tensor(name, tensor, where=where)
        self._active = prev

    def log_tensor(self, name: str, tensor: Any, where: str = "") -> None:
        if not self.cfg.enabled or not self._active:
            return
        if tensor is None:
            return
        if not torch.is_tensor(tensor):
            # allow logging python scalars as tiny "scalar"
            if isinstance(tensor, (int, float)):
                print(
                    f"[tensor] where={where} {name} scalar={tensor}",
                    flush=True,
                )
            return

        t = tensor
        device = str(t.device)
        shape = tuple(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        numel = t.numel()
        try:
            itemsize = t.element_size()
            nbytes = numel * itemsize
        except Exception:
            nbytes = -1

        parts = [
            f"[tensor] where={where}",
            f"{name}",
            f"shape={shape}",
            f"dtype={dtype}",
            f"device={device}",
            f"numel={numel}",
            f"bytes={nbytes}",
        ]

        if self.cfg.values and t.is_floating_point() and numel > 0:
            try:
                with torch.no_grad():
                    tf = t.detach().float()
                    parts.append(f"mean={tf.mean().item():.6g}")
                    parts.append(f"std={tf.std().item():.6g}")
                    parts.append(f"min={tf.min().item():.6g}")
                    parts.append(f"max={tf.max().item():.6g}")
            except Exception as exc:
                parts.append(f"stats_error={exc!r}")

        if self.cfg.full and numel > 0 and numel <= self.cfg.full_max:
            try:
                with torch.no_grad():
                    parts.append(f"data={t.detach().cpu().flatten().tolist()}")
            except Exception as exc:
                parts.append(f"data_error={exc!r}")

        print(" ".join(parts), flush=True)

    def log_param(self, name: str, param: torch.nn.Parameter, where: str = "params") -> None:
        """Log a parameter once (ignores step gating if master switch on)."""
        if not self.cfg.enabled:
            return
        # temporarily force active for one-shot param dump
        prev = self._active
        self._active = True
        self.log_tensor(name, param.data, where=where)
        self._active = prev


_LOGGER = TensorLogger()


def configure_tensor_logger(cfg: TensorLogConfig | None = None, **kwargs: Any) -> TensorLogger:
    if cfg is None:
        cfg = TensorLogConfig(**kwargs)
    _LOGGER.configure(cfg)
    return _LOGGER


def get_tensor_logger() -> TensorLogger:
    return _LOGGER


def log_tensor(name: str, tensor: Any, where: str = "") -> None:
    _LOGGER.log_tensor(name, tensor, where=where)


def begin_step(it: int) -> None:
    _LOGGER.begin_step(it)


def should_log_model() -> bool:
    return _LOGGER.should_log_model()


def should_log_data() -> bool:
    return _LOGGER.should_log_data()


def should_log_block(layer_idx: int) -> bool:
    return _LOGGER.should_log_block(layer_idx)


def tensor_log_config_from_train(train_cfg: Any) -> TensorLogConfig:
    """Build TensorLogConfig from a TrainConfig-like object."""
    return TensorLogConfig(
        enabled=bool(getattr(train_cfg, "tensor_log", False)),
        every=int(getattr(train_cfg, "tensor_log_every", 1)),
        max_steps=int(getattr(train_cfg, "tensor_log_max_steps", 5)),
        model=bool(getattr(train_cfg, "tensor_log_model", True)),
        data=bool(getattr(train_cfg, "tensor_log_data", True)),
        values=bool(getattr(train_cfg, "tensor_log_values", False)),
        full=bool(getattr(train_cfg, "tensor_log_full", False)),
        full_max=int(getattr(train_cfg, "tensor_log_full_max", 32)),
        blocks=str(getattr(train_cfg, "tensor_log_blocks", "first")),
    )
