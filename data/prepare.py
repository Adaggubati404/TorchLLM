"""Encode raw text/jsonl into uint16 memmap train/val token files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# allow `python -m data.prepare` from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torchllm.tokenizer import Tokenizer


def iter_texts(input_path: Path):
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(
            list(input_path.rglob("*.txt")) + list(input_path.rglob("*.jsonl"))
        )
    if not paths:
        raise FileNotFoundError(f"No .txt or .jsonl files under {input_path}")

    for path in paths:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        yield obj
                    elif "text" in obj:
                        yield obj["text"]
                    elif "content" in obj:
                        yield obj["content"]
                    else:
                        # fallback: dump whole object
                        yield json.dumps(obj, ensure_ascii=False)
        else:
            yield path.read_text(encoding="utf-8")


def encode_corpus(input_path: Path, tokenizer: Tokenizer) -> np.ndarray:
    ids: list[int] = []
    for text in tqdm(list(iter_texts(input_path)), desc="encoding"):
        # separate documents with an endoftext token when present in GPT-2
        piece = tokenizer.encode_ordinary(text)
        if not piece:
            continue
        ids.extend(piece)
        # 50256 is <|endoftext|> in GPT-2
        ids.append(tokenizer.enc.eot_token)
    if not ids:
        raise ValueError(f"No tokens produced from {input_path}")
    arr = np.array(ids, dtype=np.uint16)
    return arr


def split_and_write(tokens: np.ndarray, out_dir: Path, val_ratio: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(tokens)
    if n < 100:
        # tiny smoke corpora: fixed ~10% or at least 1 token in val
        n_val = max(1, n // 10)
    else:
        n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    if n_train < 1:
        raise ValueError("Not enough tokens after val split")

    train = tokens[:n_train]
    val = tokens[n_train:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    train.tofile(train_path)
    val.tofile(val_path)

    meta = {
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "total_tokens": int(n),
        "dtype": "uint16",
        "vocab": "gpt2",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {train_path} ({n_train:,} tokens)")
    print(f"Wrote {val_path} ({n_val:,} tokens)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare token memmaps for TorchLLM")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input .txt/.jsonl file or directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for train.bin / val.bin",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Fraction of tokens reserved for validation",
    )
    args = parser.parse_args(argv)

    tokenizer = Tokenizer()
    tokens = encode_corpus(Path(args.input), tokenizer)
    split_and_write(tokens, Path(args.out), args.val_ratio)


if __name__ == "__main__":
    main()
