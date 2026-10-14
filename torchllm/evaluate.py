"""Evaluate a checkpoint: mean CE loss and perplexity on val (or train) data."""

from __future__ import annotations

import argparse
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torchllm.config import config_from_dict, load_config
from torchllm.dataset import get_batch, load_bin
from torchllm.model import GPT
from torchllm.utils import get_autocast_dtype, get_device, load_checkpoint


@torch.no_grad()
def evaluate_split(
    model: GPT,
    data,
    *,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
    ctx,
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate TorchLLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML; defaults to config stored in checkpoint",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args(argv)

    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, map_location=device)

    if args.config:
        config = load_config(args.config)
    else:
        config = config_from_dict(ckpt["config"])

    model = GPT(config.model)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    bin_path = Path(args.data_dir) / f"{args.split}.bin"
    data = load_bin(bin_path)
    batch_size = args.batch_size or config.train.batch_size

    autocast_dtype = get_autocast_dtype(device, config.train.dtype)
    if device.type == "cuda" and autocast_dtype is not None:
        ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        ctx = nullcontext()

    loss = evaluate_split(
        model,
        data,
        batch_size=batch_size,
        block_size=config.model.block_size,
        eval_iters=args.eval_iters,
        device=device,
        ctx=ctx,
    )
    ppl = math.exp(min(loss, 20.0))
    print(f"split={args.split} loss={loss:.4f} ppl={ppl:.2f}")
    print(f"checkpoint iter={ckpt.get('iter')} best_val_loss={ckpt.get('best_val_loss')}")


if __name__ == "__main__":
    main()
