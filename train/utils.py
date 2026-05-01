"""Helpers shared by train.py and lr_sweep.py."""
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, peak: float, end_frac: float = 0.1) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return peak * end_frac
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak * (end_frac + (1.0 - end_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


class TokenDataset:
    """Memmap-backed random sampler. Each call returns (x, y) of shape (B, T)."""
    def __init__(self, bin_path: Path, block_size: int, dtype=np.uint16):
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        self.dtype = dtype
        self.data = np.memmap(self.bin_path, dtype=dtype, mode="r")
        if self.data.size <= block_size + 1:
            raise RuntimeError(f"{bin_path} has only {self.data.size} tokens; need > {block_size}")

    def sample(self, batch_size: int, device: torch.device, generator: np.random.Generator):
        n = self.data.size
        starts = generator.integers(0, n - self.block_size - 1, size=batch_size)
        xs = np.stack([np.asarray(self.data[s : s + self.block_size]) for s in starts]).astype(np.int64)
        ys = np.stack([np.asarray(self.data[s + 1 : s + 1 + self.block_size]) for s in starts]).astype(np.int64)
        return torch.from_numpy(xs).to(device, non_blocking=True), torch.from_numpy(ys).to(device, non_blocking=True)


def split_param_groups(model, weight_decay: float):
    """Match nanoGPT's convention: weight decay applies to >=2D weights only (linear + embedding).
    LayerNorm weights, biases, and 1D parameters are excluded."""
    decay, no_decay = [], []
    seen = set()
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Throughput:
    def __init__(self):
        self.t0 = time.time()
        self.tokens = 0
    def add(self, n):
        self.tokens += n
    def rate(self) -> float:
        dt = time.time() - self.t0
        return self.tokens / dt if dt > 0 else 0.0


def autocast_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]
