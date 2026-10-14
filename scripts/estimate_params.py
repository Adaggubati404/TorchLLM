#!/usr/bin/env python3
"""Print parameter count for a model config (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torchllm.config import load_config
from torchllm.model import GPT
from torchllm.utils import count_parameters, format_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate model parameter count")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "gpt2_small.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = GPT(config.model)
    n = count_parameters(model)
    n_non_pos = model.get_num_params(non_embedding=True)
    print(f"config: {args.config}")
    print(f"  n_layer={config.model.n_layer} n_embd={config.model.n_embd} "
          f"n_head={config.model.n_head} block_size={config.model.block_size} "
          f"vocab_size={config.model.vocab_size}")
    print(f"  total params: {n:,} ({format_params(n)})")
    print(f"  non-pos-embed params: {n_non_pos:,} ({format_params(n_non_pos)})")


if __name__ == "__main__":
    main()
