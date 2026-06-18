# tara1.1-pi

`tara1.1-pi` is a small continuous-pretraining experiment for modeling the digits
of pi as a sequence. It is not a language model; it trains an autoregressive base
model over digit tokens.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

PyTorch is already installed on this machine with CUDA support.

## Generate A Pi Dataset

This creates `data/pi.txt` with `3.` followed by pi digits.

```powershell
python scripts/make_pi_dataset.py --digits 1000000 --out data/pi.txt
```

For a quick smoke test, use fewer digits:

```powershell
python scripts/make_pi_dataset.py --digits 10000 --out data/pi.txt
```

## Train Continuously

The trainer supports resume checkpoints, so you can keep pretraining from the
latest checkpoint.

```powershell
python scripts/train.py --data data/pi.txt --out runs/tara1.1-pi --compile
```

Resume:

```powershell
python scripts/train.py --data data/pi.txt --out runs/tara1.1-pi --resume runs/tara1.1-pi/last.pt
```

Useful small test run:

```powershell
python scripts/train.py --data data/pi.txt --out runs/debug --max-steps 100 --batch-size 8 --grad-accum 2 --block-size 128 --n-layer 4 --n-head 4 --n-kv-head 2 --n-embd 256
```

## Sample

```powershell
python scripts/sample.py --checkpoint runs/tara1.1-pi/last.pt --prompt "3.14159" --tokens 100
```

## Model

Default architecture:

- decoder-only Transformer
- vocabulary: `0-9` plus `.`
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward blocks
- grouped-query attention
- mixed precision training on CUDA
- gradient accumulation
- warmup plus cosine learning-rate decay
- objective: next-token prediction
- checkpoint files: `last.pt` and periodic `step_*.pt`

Because the training data is only pi, this model learns one numeric stream. It
will not learn general math, reasoning, or natural language from this dataset.
