"""Memmap-backed random-window language modeling dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapLMDataset(Dataset):
    """Random contiguous windows from a 1D uint16 token stream."""

    def __init__(self, bin_path: str | Path, block_size: int) -> None:
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Token file not found: {self.bin_path}")
        self.block_size = block_size
        # store path and reload lazily so workers and pickling stay safe
        self._data: np.memmap | None = None
        data = self._open()
        if len(data) < block_size + 1:
            raise ValueError(
                f"{self.bin_path} has only {len(data)} tokens; "
                f"need at least block_size+1={block_size + 1}"
            )
        # number of valid start positions
        self._len = len(data) - block_size

    def _open(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        return self._data

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ignore Dataset index; sample a random window for infinite-data style
        data = self._open()
        i = torch.randint(0, self._len, (1,)).item()
        chunk = torch.from_numpy(data[i : i + self.block_size + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_bin(path: str | Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of windows without DataLoader (simple / low-overhead path)."""
    n = len(data) - block_size
    if n <= 0:
        raise ValueError("Token stream too short for block_size")
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
