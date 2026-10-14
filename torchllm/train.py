"""Training entrypoint for TorchLLM."""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from torchllm.config import Config, load_config
from torchllm.dataset import get_batch, load_bin
from torchllm.model import GPT
from torchllm.tensor_log import (
    begin_step,
    configure_tensor_logger,
    get_tensor_logger,
    log_tensor,
    should_log_data,
    tensor_log_config_from_train,
)
from torchllm.utils import (
    MetricsLogger,
    configure_optimizer,
    count_parameters,
    format_params,
    get_autocast_dtype,
    get_device,
    get_lr,
    load_checkpoint,
    needs_grad_scaler,
    prune_step_checkpoints,
    save_checkpoint,
    set_seed,
)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data,
    val_data,
    *,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
    ctx,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def build_model(config: Config, device: torch.device) -> GPT:
    model = GPT(config.model)
    model.to(device)
    return model


def _log_model_params_once(model: GPT) -> None:
    tlog = get_tensor_logger()
    if not tlog.cfg.enabled:
        return
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    tlog.log_param("wte.weight", raw.transformer.wte.weight, where="params")
    tlog.log_param("wpe.weight", raw.transformer.wpe.weight, where="params")
    if len(raw.transformer.h) > 0:
        tlog.log_param(
            "h.0.attn.c_attn.weight",
            raw.transformer.h[0].attn.c_attn.weight,
            where="params",
        )
        tlog.log_param(
            "h.0.mlp.c_fc.weight",
            raw.transformer.h[0].mlp.c_fc.weight,
            where="params",
        )
    tlog.log_param("lm_head.weight", raw.lm_head.weight, where="params")


def train(config: Config, data_dir: Path, resume: Path | None = None) -> None:
    tcfg = config.train
    mcfg = config.model
    set_seed(tcfg.seed)
    device = get_device()
    print(f"device: {device}")

    configure_tensor_logger(tensor_log_config_from_train(tcfg))

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected train.bin and val.bin in {data_dir}. "
            "Run: python -m data.prepare --input ... --out ..."
        )
    train_data = load_bin(train_path)
    val_data = load_bin(val_path)
    print(
        f"data: train={len(train_data):,} tokens, val={len(val_data):,} tokens, "
        f"block_size={mcfg.block_size}"
    )
    if len(train_data) < mcfg.block_size + 1:
        raise ValueError(
            f"train.bin too short ({len(train_data)}) for block_size {mcfg.block_size}"
        )
    if len(val_data) < mcfg.block_size + 1:
        raise ValueError(
            f"val.bin too short ({len(val_data)}) for block_size {mcfg.block_size}. "
            "Use more data or a smaller block_size (e.g. smoke config)."
        )

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricsLogger(out_dir / "metrics.jsonl")

    model = build_model(config, device)
    n_params = count_parameters(model)
    print(f"model params: {n_params:,} ({format_params(n_params)})")
    _log_model_params_once(model)

    optimizer = configure_optimizer(
        model,
        weight_decay=tcfg.weight_decay,
        lr=tcfg.lr,
        betas=(tcfg.beta1, tcfg.beta2),
        device_type=device.type,
    )

    start_iter = 0
    best_val_loss = float("inf")

    if resume is not None:
        print(f"resuming from {resume}")
        ckpt = load_checkpoint(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt.get("iter", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if "config" in ckpt:
            pass

    if tcfg.compile and hasattr(torch, "compile") and device.type == "cuda":
        print("compiling model with torch.compile ...")
        model = torch.compile(model)  # type: ignore[assignment]

    autocast_dtype = get_autocast_dtype(device, tcfg.dtype)
    use_scaler = needs_grad_scaler(device, autocast_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if device.type == "cuda" and autocast_dtype is not None:
        ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    elif device.type == "mps" and autocast_dtype is not None:
        try:
            ctx = torch.amp.autocast(device_type="mps", dtype=autocast_dtype)
        except Exception:
            print("warning: MPS autocast unavailable; falling back to full precision")
            ctx = nullcontext()
            autocast_dtype = None
    else:
        ctx = nullcontext()

    ac_name = "none" if autocast_dtype is None else str(autocast_dtype).replace("torch.", "")
    print(f"dtype config={tcfg.dtype} autocast={ac_name}")

    if device.type == "mps" and tcfg.batch_size * mcfg.block_size > 2048:
        print(
            "warning: large micro-batch on MPS may OOM "
            f"(B={tcfg.batch_size} T={mcfg.block_size}). "
            "Prefer configs/gpt2_small_mps.yaml "
            "(B=1, accum=64, float16) or pass --batch-size 1 --grad-accum-steps 64 --dtype float16"
        )

    tokens_per_step = tcfg.batch_size * mcfg.block_size * tcfg.grad_accum_steps
    print(
        f"tokens/step={tokens_per_step:,} "
        f"(B={tcfg.batch_size} T={mcfg.block_size} accum={tcfg.grad_accum_steps})"
    )
    print(
        f"planned total tokens ≈ {tcfg.max_iters * tokens_per_step:,} "
        f"over {tcfg.max_iters:,} steps"
    )

    model.train()
    t0 = time.time()
    running_t0 = time.time()

    for it in range(start_iter, tcfg.max_iters):
        begin_step(it)
        lr = get_lr(
            it,
            peak_lr=tcfg.lr,
            min_lr=tcfg.min_lr,
            warmup_iters=tcfg.warmup_iters,
            max_iters=tcfg.max_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(tcfg.grad_accum_steps):
            x, y = get_batch(train_data, tcfg.batch_size, mcfg.block_size, device)
            if micro == 0 and should_log_data():
                log_tensor("x", x, where="train.batch")
                log_tensor("y", y, where="train.batch")
            with ctx:
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item()

        if should_log_data():
            log_tensor("loss_accum", torch.tensor(loss_accum), where="train.step")

        if tcfg.grad_clip > 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if it % tcfg.log_interval == 0 or it == tcfg.max_iters - 1:
            dt = time.time() - running_t0
            running_t0 = time.time()
            tps = tokens_per_step / dt if dt > 0 else 0.0
            steps_spanned = tcfg.log_interval if it > start_iter else 1
            if it % tcfg.log_interval == 0 and it > start_iter:
                tps = (tokens_per_step * steps_spanned) / dt if dt > 0 else 0.0
            metrics.log(
                {
                    "event": "train",
                    "iter": it,
                    "train_loss": loss_accum,
                    "lr": lr,
                    "tokens_per_sec": tps,
                    "elapsed": time.time() - t0,
                }
            )

        do_eval = (it % tcfg.eval_interval == 0) or (it == tcfg.max_iters - 1)
        if do_eval:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size=tcfg.batch_size,
                block_size=mcfg.block_size,
                eval_iters=tcfg.eval_iters,
                device=device,
                ctx=ctx,
            )
            val_loss = losses["val"]
            val_ppl = math.exp(min(val_loss, 20.0))
            metrics.log(
                {
                    "event": "eval",
                    "iter": it,
                    "train_loss_est": losses["train"],
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "best_val_loss": min(best_val_loss, val_loss),
                }
            )

            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            payload = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": it + 1,
                "best_val_loss": min(best_val_loss, val_loss),
                "config": config.to_dict(),
                "n_params": n_params,
            }
            save_checkpoint(out_dir / "ckpt_last.pt", payload)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                payload["best_val_loss"] = best_val_loss
                save_checkpoint(out_dir / "ckpt_best.pt", payload)
                print(f"new best val_loss={best_val_loss:.4f} @ iter {it}")

        if it > 0 and it % tcfg.save_interval == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            payload = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": it + 1,
                "best_val_loss": best_val_loss,
                "config": config.to_dict(),
                "n_params": n_params,
            }
            step_path = out_dir / f"ckpt_step_{it}.pt"
            save_checkpoint(step_path, payload)
            prune_step_checkpoints(out_dir, tcfg.keep_last_k)

    print(f"done. best_val_loss={best_val_loss:.4f}")
    print(f"checkpoints in {out_dir.resolve()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train TorchLLM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with train.bin and val.bin",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Override max_iters from config",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override out_dir from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override micro-batch size from config",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps from config",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Override train dtype (e.g. float16, bfloat16, float32)",
    )
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
        help="Override max training steps that emit tensor logs (0=unlimited)",
    )
    parser.add_argument(
        "--tensor-log-values",
        action="store_true",
        default=None,
        help="Include mean/std/min/max in tensor logs",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.max_iters is not None:
        config.train.max_iters = args.max_iters
    if args.out_dir is not None:
        config.train.out_dir = args.out_dir
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.train.grad_accum_steps = args.grad_accum_steps
    if args.dtype is not None:
        config.train.dtype = args.dtype
    if args.tensor_log is not None:
        config.train.tensor_log = args.tensor_log
    if args.tensor_log_max_steps is not None:
        config.train.tensor_log_max_steps = args.tensor_log_max_steps
    if args.tensor_log_values is not None:
        config.train.tensor_log_values = True

    resume = Path(args.resume) if args.resume else None
    train(config, Path(args.data_dir), resume=resume)


if __name__ == "__main__":
    main()
