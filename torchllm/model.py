"""GPT-style decoder-only language model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchllm.config import ModelConfig
from torchllm.tensor_log import log_tensor, should_log_block, should_log_model
from torchllm.tensor_log import get_tensor_logger


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        do_log = should_log_block(self.layer_idx)
        where = f"CausalSelfAttention.L{self.layer_idx}"
        if do_log:
            log_tensor("x", x, where=where)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if do_log:
            log_tensor("qkv", qkv, where=where)
            log_tensor("q", q, where=where)
            log_tensor("k", k, where=where)
            log_tensor("v", v, where=where)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if do_log:
            log_tensor("y_merged", y, where=where)

        out = self.resid_dropout(self.c_proj(y))
        if do_log:
            log_tensor("out", out, where=where)
        return out


class MLP(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        do_log = should_log_block(self.layer_idx)
        where = f"MLP.L{self.layer_idx}"
        if do_log:
            log_tensor("x", x, where=where)
        h = self.c_fc(x)
        if do_log:
            log_tensor("c_fc", h, where=where)
        h = self.gelu(h)
        out = self.dropout(self.c_proj(h))
        if do_log:
            log_tensor("out", out, where=where)
        return out


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, layer_idx=layer_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        do_log = should_log_block(self.layer_idx)
        where = f"Block.L{self.layer_idx}"
        if do_log:
            log_tensor("x_in", x, where=where)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if do_log:
            log_tensor("x_out", x, where=where)
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config, layer_idx=i) for i in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        # residual path scaling (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(
                f"Sequence length {t} exceeds block_size {self.config.block_size}"
            )

        do_log = should_log_model()
        where = "GPT.forward"
        if do_log:
            log_tensor("idx", idx, where=where)
            if targets is not None:
                log_tensor("targets", targets, where=where)

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        if do_log:
            log_tensor("pos", pos, where=where)
            log_tensor("tok_emb", tok_emb, where=where)
            log_tensor("pos_emb", pos_emb, where=where)
            log_tensor("x_after_drop", x, where=where)

        for i, block in enumerate(self.transformer.h):
            x = block(x)
            # residual stream after selected blocks only when model logging active
            if do_log and should_log_block(i):
                log_tensor(f"x_after_block_{i}", x, where=where)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if do_log:
            log_tensor("x_after_ln_f", x, where=where)
            log_tensor("logits", logits, where=where)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            if do_log:
                log_tensor("loss", loss, where=where)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        tlog = get_tensor_logger()
        for _ in range(max_new_tokens):
            do_log = tlog.should_log_generate_token()
            where = "GPT.generate"
            if do_log:
                log_tensor("idx", idx, where=where)

            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            if do_log:
                log_tensor("idx_cond", idx_cond, where=where)

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if do_log:
                log_tensor("logits_last", logits, where=where)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if do_log:
                log_tensor("idx_next", idx_next, where=where)
            idx = torch.cat((idx, idx_next), dim=1)
        tlog.force_log_tensor("idx_final", idx, where="GPT.generate")
        return idx

    def get_num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n


@dataclass
class GPTConfig:
    """Thin alias kept for scripts that expect a simple config object."""

    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    block_size: int = 1024
    vocab_size: int = 50304
    dropout: float = 0.0
    bias: bool = True

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            bias=self.bias,
        )
