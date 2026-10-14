# TorchLLM

GPT-2 Small–class (~124M parameters) decoder-only language model in pure PyTorch.

## Features

- Config-driven model (`gpt2_small` ≈ 124M, plus a tiny `smoke` config)
- GPT-2 BPE via `tiktoken`, vocab padded to 50304
- Memmap data pipeline (`train.bin` / `val.bin`)
- AdamW with decay/no-decay groups, linear warmup + cosine LR
- Gradient accumulation, mixed precision (BF16/FP16 when available)
- Checkpoints: `ckpt_last.pt`, `ckpt_best.pt`, stepped saves + resume
- Eval (loss / perplexity) and temperature + top-k generation
- Optional tensor debug logs (shape, dtype, device) via config/CLI

## Install

```bash
cd TorchLLM
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Requires Python 3.10+ and a working PyTorch install for your platform ([pytorch.org](https://pytorch.org)).

## Quick start (smoke train)

Tokenize the sample corpus, train a tiny model, evaluate, generate:

```bash
# 1) Prepare tokens
python -m data.prepare --input data/sample --out data/sample_bin --val-ratio 0.1

# 2) Estimate params for a config (optional)
python scripts/estimate_params.py --config configs/gpt2_small.yaml
python scripts/estimate_params.py --config configs/smoke.yaml

# 3) Smoke train (CPU/MPS/CUDA friendly)
python -m torchllm.train --config configs/smoke.yaml --data-dir data/sample_bin

# 4) Evaluate best checkpoint
python -m torchllm.evaluate \
  --checkpoint checkpoints/smoke/ckpt_best.pt \
  --data-dir data/sample_bin

# 5) Generate
python -m torchllm.generate \
  --checkpoint checkpoints/smoke/ckpt_best.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 50
```

Or: `bash scripts/train.sh configs/smoke.yaml data/sample_bin`

## Full ~124M pretrain

Architecture defaults (`configs/gpt2_small.yaml` for CUDA; `configs/gpt2_small_mps.yaml` for Mac):

| | |
|---|---|
| layers / dim / heads | 12 / 768 / 12 |
| context | 1024 |
| vocab | 50304 (GPT-2 + pad) |
| peak LR | 3e-4 (warmup 2k → cosine to 3e-5) |
| batch (CUDA) | `B=8`, accum `8` → **65,536 tokens/step** |
| batch (MPS) | `B=1`, accum `64` → **65,536 tokens/step** (same effective size) |
| default steps | 100,000 (~6.6B tokens at default batch) |

Prepare a larger corpus (directory of `.txt` / `.jsonl` with a `text` field), then train:

```bash
python -m data.prepare --input /path/to/corpus --out data/my_corpus_bin --val-ratio 0.05

# CUDA / high VRAM
python -m torchllm.train \
  --config configs/gpt2_small.yaml \
  --data-dir data/my_corpus_bin

# Apple Silicon (MPS) — required to avoid OOM at full context
python -m torchllm.train \
  --config configs/gpt2_small_mps.yaml \
  --data-dir data/my_corpus_bin

# Or override the CUDA config for MPS:
python -m torchllm.train \
  --config configs/gpt2_small.yaml \
  --data-dir data/my_corpus_bin \
  --batch-size 1 --grad-accum-steps 64 --dtype float16

# resume
python -m torchllm.train \
  --config configs/gpt2_small_mps.yaml \
  --data-dir data/my_corpus_bin \
  --resume checkpoints/gpt2_small_mps/ckpt_last.pt
```

### Apple Silicon / MPS OOM

`configs/gpt2_small.yaml` uses `B=8 × T=1024` with BF16. On MPS, BF16 falls back to **float16** automatically, but the large micro-batch still OOMs (~20 GiB budget). Use **`configs/gpt2_small_mps.yaml`** (`B=1`, accum `64`, float16). If memory is still tight, lower `block_size` (e.g. 512) in a copy of that config.

### Hardware notes

| VRAM | Suggestion |
|---|---|
| 8–12 GB | `batch_size: 1–2`, raise `grad_accum_steps` |
| 16–24 GB CUDA | defaults (`B=8`, accum `8`) with BF16 |
| MPS (Apple) | `configs/gpt2_small_mps.yaml` |
| 40+ GB | larger `B`, fewer accum steps; aim 0.25–0.5M tokens/step |
| CPU / small data | `configs/smoke.yaml` only |

If you OOM, lower `train.batch_size` first (CLI: `--batch-size 1`).

### Tensor debug logging (shape / dtype / device)

Off by default. When enabled, logs tensor **shape**, **dtype**, **device location**, **numel**, and **bytes** at major train/model sites (`[tensor] ...` lines).

YAML (`train` section):

```yaml
tensor_log: true
tensor_log_every: 1          # log on steps where it % every == 0
tensor_log_max_steps: 5      # stop after N training steps (0 = unlimited)
tensor_log_model: true       # GPT / attention / MLP / blocks
tensor_log_data: true        # train batches x/y
tensor_log_values: false     # optional mean/std/min/max
tensor_log_full: false       # dump small tensor bodies
tensor_log_full_max: 32
tensor_log_blocks: first     # first | all  (layer span for block logs)
```

CLI:

```bash
# short smoke run with tensor traces
python -m torchllm.train \
  --config configs/smoke.yaml \
  --data-dir data/sample_bin \
  --max-iters 2 \
  --tensor-log \
  --tensor-log-max-steps 2

# with activation stats
python -m torchllm.train ... --tensor-log --tensor-log-values

# generate with logs
python -m torchllm.generate \
  --checkpoint checkpoints/smoke/ckpt_best.pt \
  --tensor-log \
  --tensor-log-max-steps 3 \
  --prompt "Once upon a time"
```

Example line:

```text
[tensor] where=GPT.forward idx shape=(4, 64) dtype=int64 device=mps:0 numel=256 bytes=2048
```

**Warning:** logging every layer on every step is expensive. Keep `tensor_log_max_steps` low and prefer `tensor_log_blocks: first` for routine debugging.

### Expected loss (directional)

- Random init ≈ `ln(50304) ≈ 10.8`
- Sample smoke overfit: train loss should fall clearly within a few hundred steps
- Open-web pretrain of a ~100M model after billions of tokens often lands around val CE **~3.0–3.5** (data-dependent)

A useful 100M model needs **billions of tokens**. The sample path only proves the pipeline.

## Project layout

```
configs/           # gpt2_small.yaml, gpt2_small_mps.yaml, smoke.yaml
data/prepare.py    # raw text → train.bin / val.bin
data/sample/       # tiny corpus for smoke tests
torchllm/
  model.py         # GPT decoder
  train.py         # training loop
  evaluate.py      # val loss / ppl
  generate.py      # sampling
  tensor_log.py    # configurable tensor shape/device debug logs
scripts/estimate_params.py
```

## License

MIT (project scaffolding; you own your models and data).
