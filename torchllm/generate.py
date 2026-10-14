"""Sample text from a trained TorchLLM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torchllm.config import config_from_dict, load_config
from torchllm.model import GPT
from torchllm.tensor_log import (
    configure_tensor_logger,
    get_tensor_logger,
    tensor_log_config_from_train,
)
from torchllm.tokenizer import Tokenizer
from torchllm.utils import get_device, load_checkpoint, set_seed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate text with TorchLLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML; defaults to config stored in checkpoint",
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--tensor-log",
        dest="tensor_log",
        action="store_true",
        default=None,
        help="Enable tensor shape/device debug logs",
    )
    parser.add_argument(
        "--no-tensor-log",
        dest="tensor_log",
        action="store_false",
        help="Disable tensor debug logs",
    )
    parser.add_argument(
        "--tensor-log-max-steps",
        type=int,
        default=None,
        help="Max generate steps that emit tensor logs (default 5)",
    )
    parser.add_argument(
        "--tensor-log-values",
        action="store_true",
        default=None,
        help="Include mean/std/min/max in tensor logs",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, map_location=device)

    if args.config:
        config = load_config(args.config)
    else:
        config = config_from_dict(ckpt["config"])

    if args.tensor_log is not None:
        config.train.tensor_log = args.tensor_log
    if args.tensor_log_max_steps is not None:
        config.train.tensor_log_max_steps = args.tensor_log_max_steps
    if args.tensor_log_values is not None:
        config.train.tensor_log_values = True

    tcfg = tensor_log_config_from_train(config.train)
    configure_tensor_logger(tcfg)
    tlog = get_tensor_logger()
    tlog.enable_for_run()

    model = GPT(config.model)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tokenizer = Tokenizer()
    start_ids = tokenizer.encode_ordinary(args.prompt)
    if not start_ids:
        start_ids = [tokenizer.enc.eot_token]
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    if tlog.cfg.enabled:
        tlog.force_log_tensor("prompt_idx", idx, where="generate")

    with torch.no_grad():
        out = model.generate(
            idx,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )
    tokens = out[0].tolist()
    text = tokenizer.decode(tokens)
    print(text)


if __name__ == "__main__":
    main()
