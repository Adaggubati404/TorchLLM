"""GPT-2 BPE tokenizer via tiktoken, with padded vocab size."""

from __future__ import annotations

import tiktoken

# GPT-2 native size; we pad model vocab to a multiple of 64 for kernel alignment.
GPT2_VOCAB_SIZE = 50257
PADDED_VOCAB_SIZE = 50304


class Tokenizer:
    def __init__(self) -> None:
        self.enc = tiktoken.get_encoding("gpt2")
        self.n_vocab = GPT2_VOCAB_SIZE
        self.padded_vocab_size = PADDED_VOCAB_SIZE

    def encode(self, text: str, *, allowed_special: str | set[str] = "all") -> list[int]:
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, ids: list[int]) -> str:
        # clamp any padded-range ids (should not appear in real data)
        clean = [i for i in ids if 0 <= i < self.n_vocab]
        return self.enc.decode(clean)

    def encode_ordinary(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)
