"""Shared helpers: device, seed, LR schedule, checkpoints, param counts."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_dtype(device: torch.device, dtype_name: str) -> torch.dtype | None:
    """Return autocast dtype, or None when autocast should be disabled."""
    name = dtype_name.lower()
    if device.type == "cpu":
        return None
    if name in ("float32", "fp32"):
        return None
    if name in ("bfloat16", "bf16"):
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # MPS has no reliable BF16; fall back to FP16 (not full FP32)
        if device.type in ("cuda", "mps"):
            return torch.float16
        return None
    if name in ("float16", "fp16", "half"):
        if device.type in ("cuda", "mps"):
            return torch.float16
        return None
    raise ValueError(f"Unknown dtype: {dtype_name}")


def needs_grad_scaler(device: torch.device, autocast_dtype: torch.dtype | None) -> bool:
    return device.type == "cuda" and autocast_dtype == torch.float16


def get_lr(
    it: int,
    *,
    peak_lr: float,
    min_lr: float,
    warmup_iters: int,
    max_iters: int,
) -> float:
    """Linear warmup then cosine decay to min_lr (iteration-based)."""
    if it < warmup_iters:
        # avoid zero lr at step 0
        return peak_lr * (it + 1) / max(1, warmup_iters)
    if it >= max_iters:
        return min_lr
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def configure_optimizer(
    model: torch.nn.Module,
    *,
    weight_decay: float,
    lr: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.Optimizer:
    """AdamW with decay on 2D params only (matmuls / embeddings)."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # weight tying: same Parameter can appear under multiple names
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        # biases and layernorm weights are 1D; matmuls and embeddings are 2D+
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused AdamW when available on CUDA
    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == "cuda"
    extra = {"fused": True} if use_fused else {}
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, **extra)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def prune_step_checkpoints(out_dir: Path, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    steps = sorted(out_dir.glob("ckpt_step_*.pt"))
    for old in steps[:-keep_last_k]:
        old.unlink(missing_ok=True)


class MetricsLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        record = {**record, "time": time.time()}
        line = json.dumps(record)
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
