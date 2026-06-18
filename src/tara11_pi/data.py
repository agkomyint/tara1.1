from __future__ import annotations

from pathlib import Path

import torch

from tara11_pi.tokenizer import DigitTokenizer


def load_encoded(path: Path, tokenizer: DigitTokenizer) -> torch.Tensor:
    text = path.read_text(encoding="utf-8").strip()
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)


class PiSequenceDataset:
    def __init__(self, data: torch.Tensor, block_size: int, batch_size: int):
        if data.ndim != 1:
            raise ValueError("data must be a 1D token tensor")
        if len(data) <= block_size + 1:
            raise ValueError("dataset must be longer than block_size + 1")
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size

    def random_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(self.data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([self.data[i : i + self.block_size] for i in starts])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in starts])
        return x, y
