from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from tara11_pi.data import PiSequenceDataset, load_encoded
from tara11_pi.model import GPT, GPTConfig
from tara11_pi.tokenizer import DigitTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous pretraining for tara1.1-pi.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/tara1.1-pi"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-kv-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    return parser.parse_args()


def save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, step: int, config: GPTConfig) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": config.__dict__,
        },
        path,
    )


@torch.no_grad()
def estimate_loss(model: GPT, dataset: PiSequenceDataset, device: str, batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = dataset.random_batch()
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = device == "cuda"
    dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = DigitTokenizer()
    encoded = load_encoded(args.data, tokenizer)
    val_tokens = max(args.block_size + 2, int(len(encoded) * args.val_fraction))
    train_encoded = encoded[:-val_tokens]
    val_encoded = encoded[-val_tokens:]
    train_dataset = PiSequenceDataset(train_encoded, block_size=args.block_size, batch_size=args.batch_size)
    val_dataset = PiSequenceDataset(val_encoded, block_size=args.block_size, batch_size=args.batch_size)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        config = GPTConfig(**checkpoint["model_config"])
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint["model"])
        if args.compile:
            model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])

    metadata = {
        "model_name": "tara1.1-pi",
        "data": str(args.data),
        "tokens": len(encoded),
        "train_tokens": len(train_encoded),
        "val_tokens": len(val_encoded),
        "device": device,
        "dtype": str(dtype),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    model.train()
    start = time.time()
    progress = tqdm(range(step, args.max_steps), initial=step, total=args.max_steps)
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(args.grad_accum):
            x, y = train_dataset.random_batch()
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=amp_enabled):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        current_step = step + 1
        progress.set_description(f"loss {total_loss:.4f} lr {lr:.2e}")

        if current_step % args.eval_every == 0:
            eval_loss = estimate_loss(model, val_dataset, device)
            ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            elapsed = time.time() - start
            print(f"step={current_step} loss={eval_loss:.4f} ppl={ppl:.2f} elapsed={elapsed:.1f}s")

        if current_step % args.save_every == 0:
            save_checkpoint(args.out / "last.pt", model, optimizer, current_step, config)
            save_checkpoint(args.out / f"step_{current_step}.pt", model, optimizer, current_step, config)

    save_checkpoint(args.out / "last.pt", model, optimizer, args.max_steps, config)


if __name__ == "__main__":
    main()
