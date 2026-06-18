from __future__ import annotations


class DigitTokenizer:
    def __init__(self) -> None:
        self.tokens = [".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for token, i in self.stoi.items()}
        self.vocab_size = len(self.tokens)

    def encode(self, text: str) -> list[int]:
        ids = []
        for char in text:
            if char.isspace():
                continue
            if char not in self.stoi:
                raise ValueError(f"unsupported character in digit stream: {char!r}")
            ids.append(self.stoi[char])
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)
