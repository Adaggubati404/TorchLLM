# TorchLLM

A **GPT-2 Small–class decoder-only language model** (~124M parameters in its default configuration) implemented in plain PyTorch, with a complete pipeline around it: tokenization, memory-mapped data, training, evaluation, checkpointing and text generation.

> **Note:** This is a learning project built to understand the internals; it is not trained to convergence. The ~124M configuration describes the architecture; no trained weights are published.

## Why I built it

I wanted to understand how a modern language model works from first principles, following Andrej Karpathy-style material, rather than only calling a hosted API. Writing the model and the training loop myself meant every tensor had a shape, a dtype and a device I could inspect. That is why the project includes a configurable tensor logger that traces data as it flows through the model. TorchLLM follows an earlier from-scratch version I wrote in pure Java ([JavaLLM](https://github.com/Adaggubati404/JavaLLM)).

## What this taught me

- **How data moves during training.** Text becomes GPT-2 BPE token ids, is stored once as a memory-mapped file, and is sampled as windows where the target `y` is the input `x` shifted by one token.
- **Where the matrix multiplications are.** The query/key/value projections, the attention scores, the MLP layers and the LM head are all matrix multiplies, and the tensor logger shows the shape at every step.
- **How loss is calculated.** Cross-entropy between the logits and the next-token targets, and perplexity as its exponent during evaluation. An untrained model starts near ln(50304) ≈ 10.8.
- **How backpropagation drives training.** `loss.backward()` computes the gradients. Then gradient accumulation, clipping and AdamW (with warmup and cosine decay) turn them into parameter updates.
- **Why inference doesn't need backpropagation.** Evaluation and generation run under `torch.no_grad()`: only the forward pass, with no gradients stored.
- **How temperature and top-k shape output.** Temperature flattens or sharpens the next-token distribution, and top-k cuts off unlikely tokens.
- **How layer normalization helps training.** Pre-LayerNorm blocks keep each layer's inputs on a stable scale, so the residual stack trains reliably as it gets deeper.
- **Memory is a real constraint.** The same ~124M architecture needs different batch, accumulation and precision settings on CUDA and on Apple MPS.

## Features

- **Config-driven model:** YAML configs for the full `gpt2_small` model (~124M parameters), an Apple Silicon variant (`gpt2_small_mps`), and a tiny `smoke` model for quick tests.
- **GPT-2 BPE tokenization** via `tiktoken`, with the vocabulary padded from 50,257 to 50,304 to a multiple of 64.
- **Memmap data pipeline:** raw `.txt` / `.jsonl` (with a `text` field) is encoded once into `uint16` `train.bin` / `val.bin` files.
- **Training:** AdamW with decay/no-decay parameter groups, linear warmup plus cosine learning-rate decay, gradient accumulation, gradient clipping, and mixed precision (BF16/FP16 when available, with a gradient scaler where needed).
- **Device support:** CUDA, Apple MPS, or CPU, chosen automatically; optional `torch.compile` on CUDA.
- **Checkpoints:** `ckpt_last.pt`, `ckpt_best.pt`, periodic step checkpoints (keeping the last *k*), and `--resume`.
- **Evaluation:** validation loss and perplexity.
- **Generation:** temperature and top-k sampling from a checkpoint.
- **Metrics:** training metrics written to `metrics.jsonl`.
- **Tensor debug logging:** optional logs of tensor shape, dtype, device, element count and bytes at key points in training, the model and generation, plus optional value statistics.

## Architecture

```
token ids ──► token embedding (wte) + learned position embedding (wpe) ──► dropout
          ──► N × Block [ LayerNorm → causal self-attention → residual
                          LayerNorm → MLP (GELU) → residual ]
          ──► final LayerNorm ──► LM head (weights tied to wte) ──► logits
```

- Pre-LayerNorm transformer blocks with residual connections.
- Causal self-attention using PyTorch's `scaled_dot_product_attention`.
- MLP with GELU activation.
- The LM head shares its weights with the token embedding (weight tying), and residual projections use GPT-2-style initialization scaling.

| Setting | `gpt2_small` | `smoke` |
| --- | --- | --- |
| layers / embedding dim / heads | 12 / 768 / 12 | 2 / 128 / 4 |
| context length | 1024 | 64 |
| vocabulary | 50,304 | 50,304 |
| peak / min LR | 3e-4 / 3e-5 (2,000 warmup steps) | 3e-4 / 3e-5 (20 warmup steps) |
| batch × grad accumulation | 8 × 8 (CUDA); 1 × 64 on MPS (`gpt2_small_mps`) | 4 × 2 |
| default steps | 100,000 | 200 |

At context length 1024, both `gpt2_small` variants process 65,536 tokens per optimizer step.

## Getting started

**Requirements:** Python 3.10+ and a working PyTorch install for your platform ([pytorch.org](https://pytorch.org)). Other dependencies (`tiktoken`, `numpy`, `tqdm`, `PyYAML`) install with the package.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

### Quick smoke run

```bash
# 1) Tokenize the sample corpus
python -m data.prepare --input data/sample --out data/sample_bin --val-ratio 0.1

# 2) (Optional) Print the parameter count for a config
python scripts/estimate_params.py --config configs/gpt2_small.yaml

# 3) Train the tiny smoke model
python -m torchllm.train --config configs/smoke.yaml --data-dir data/sample_bin

# 4) Evaluate the best checkpoint
python -m torchllm.evaluate --checkpoint checkpoints/smoke/ckpt_best.pt --data-dir data/sample_bin

# 5) Generate text
python -m torchllm.generate --checkpoint checkpoints/smoke/ckpt_best.pt \
  --prompt "Once upon a time" --max-new-tokens 80 --temperature 0.8 --top-k 50
```

Or run training with `bash scripts/train.sh configs/smoke.yaml data/sample_bin`. After `pip install -e .`, the commands `torchllm-train`, `torchllm-evaluate` and `torchllm-generate` are also available.

### Full ~124M pretraining

```bash
python -m data.prepare --input /path/to/corpus --out data/my_corpus_bin --val-ratio 0.05

# CUDA
python -m torchllm.train --config configs/gpt2_small.yaml --data-dir data/my_corpus_bin

# Apple Silicon (MPS): batch size 1 with 64 accumulation steps to avoid running out of memory
python -m torchllm.train --config configs/gpt2_small_mps.yaml --data-dir data/my_corpus_bin

# Resume from the last checkpoint
python -m torchllm.train --config configs/gpt2_small_mps.yaml --data-dir data/my_corpus_bin \
  --resume checkpoints/gpt2_small_mps/ckpt_last.pt
```

Useful overrides: `--batch-size`, `--grad-accum-steps`, `--dtype`, `--max-iters`, `--out-dir`. If you run out of memory, lower the batch size first. A useful ~100M-parameter model needs billions of training tokens. This project intentionally stops at the bundled sample corpus, which is enough to exercise the full pipeline.

### Tensor debug logging

```bash
python -m torchllm.train --config configs/smoke.yaml --data-dir data/sample_bin \
  --max-iters 2 --tensor-log --tensor-log-max-steps 2
```

Example log line:

```text
[tensor] where=GPT.forward idx shape=(4, 64) dtype=int64 device=mps:0 numel=256 bytes=2048
```

Logging is off by default, but `configs/smoke.yaml` turns it on. YAML options include `tensor_log_every`, `tensor_log_max_steps`, `tensor_log_model`, `tensor_log_data`, `tensor_log_values` and `tensor_log_blocks` (`first` | `all`). Logging every layer on every step is expensive, so keep `tensor_log_max_steps` low.

## Project structure

```
configs/
  gpt2_small.yaml        # ~124M model, CUDA defaults
  gpt2_small_mps.yaml    # same model, memory-safe for Apple MPS
  smoke.yaml             # tiny model for quick tests
data/
  prepare.py             # raw text / jsonl → train.bin / val.bin
  sample/input.txt       # tiny sample corpus
scripts/
  estimate_params.py     # parameter count for a config
  train.sh               # training wrapper
torchllm/
  config.py              # model and training config dataclasses, YAML loading
  tokenizer.py           # GPT-2 BPE via tiktoken, padded vocab
  dataset.py             # memmap dataset and batching
  model.py               # GPT decoder: attention, MLP, blocks, generation
  train.py               # training loop, eval, checkpoints, metrics
  evaluate.py            # validation loss / perplexity
  generate.py            # sampling CLI
  tensor_log.py          # configurable tensor shape/dtype/device logging
  utils.py               # seeding, device, mixed precision, LR schedule, optimizer, checkpoints
pyproject.toml
```

## Next steps

These are not implemented yet:

- Add unit tests for the model, data pipeline and LR schedule.
- Compare outputs and loss against a reference GPT-2 implementation.

## Roadmap

This project is one step in learning the AI stack from the ground up before moving into agentic engineering. **Next: fine-tuning the open-source Qwen model and building a real agentic application on top of it.**

## License

MIT. See [LICENSE](LICENSE).
