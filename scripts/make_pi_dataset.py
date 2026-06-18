from __future__ import annotations

import argparse
from pathlib import Path

import mpmath as mp


def pi_text(digits: int) -> str:
    if digits < 1:
        raise ValueError("--digits must be at least 1")
    mp.mp.dps = digits + 5
    value = mp.nstr(mp.pi, n=digits + 2)
    return value[: digits + 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a pi digit dataset.")
    parser.add_argument("--digits", type=int, default=1_000_000, help="Digits after the decimal point.")
    parser.add_argument("--out", type=Path, default=Path("data/pi.txt"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = pi_text(args.digits)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {len(text):,} tokens to {args.out}")


if __name__ == "__main__":
    main()
