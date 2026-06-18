from __future__ import annotations

import argparse
from pathlib import Path

import mpmath as mp
import requests
from tqdm import tqdm


PI_DELIVERY_URL = "https://api.pi.delivery/v1/pi"


def pi_text(digits: int) -> str:
    if digits < 1:
        raise ValueError("--digits must be at least 1")
    mp.mp.dps = digits + 5
    value = mp.nstr(mp.pi, n=digits + 2)
    return value[: digits + 2]


def download_pi_text(digits: int, chunk_size: int) -> str:
    if digits < 1:
        raise ValueError("--digits must be at least 1")
    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")

    total = digits + 1
    parts = []
    for start in tqdm(range(0, total, chunk_size), desc="downloading pi digits"):
        count = min(chunk_size, total - start)
        response = requests.get(
            PI_DELIVERY_URL,
            params={"start": start, "numberOfDigits": count},
            timeout=60,
        )
        response.raise_for_status()
        parts.append(response.json()["content"])

    raw = "".join(parts)
    return f"{raw[0]}.{raw[1:]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a pi digit dataset.")
    parser.add_argument("--digits", type=int, default=1_000_000, help="Digits after the decimal point.")
    parser.add_argument("--out", type=Path, default=Path("data/pi.txt"))
    parser.add_argument(
        "--source",
        choices=["download", "compute"],
        default="download",
        help="Use precomputed pi digits from pi.delivery or compute locally with mpmath.",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Digits per download request.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.source == "download":
        text = download_pi_text(args.digits, args.chunk_size)
    else:
        text = pi_text(args.digits)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {len(text):,} tokens to {args.out}")


if __name__ == "__main__":
    main()
