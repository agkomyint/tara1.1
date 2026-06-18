from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tara11_pi.model import GPT, GPTConfig
from tara11_pi.tokenizer import DigitTokenizer


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a tara1.1-pi checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="3.14159")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = DigitTokenizer()
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    idx = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(idx[0].tolist()))


if __name__ == "__main__":
    main()
