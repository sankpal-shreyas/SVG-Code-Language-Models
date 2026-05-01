"""Single-GPU training loop for SP and muP runs."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.gpt import GPT, GPTConfig
from train.utils import (
    TokenDataset, Throughput, autocast_dtype,
    cosine_lr, load_yaml, split_param_groups,
)


def build_model(arch: dict, parameterization: str, base_shapes_path: Path) -> GPT:
    cfg = GPTConfig(**arch)
    if parameterization == "sp":
        cfg.attn_scale_mode = "sp"
        cfg.mup_readout = False
        return GPT(cfg)
    if parameterization == "mup":
        from model.mup_gpt import build_mup_model
        return build_mup_model(cfg, base_shapes_path=base_shapes_path)
    raise ValueError(f"unknown parameterization: {parameterization}")


def build_optimizer(model, parameterization: str, lr: float, weight_decay: float, betas):
    groups = split_param_groups(model, weight_decay)
    if parameterization == "mup":
        try:
            from mup import MuAdamW
            return MuAdamW(groups, lr=lr, betas=betas)
        except ImportError:
            from mup import MuAdam
            return MuAdam(groups, lr=lr, betas=betas)
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


@torch.no_grad()
def evaluate(model, val_ds: TokenDataset, batch_size: int, n_batches: int, device, amp_dtype, gen) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = val_ds.sample(batch_size, device, gen)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_one(
    config_path: Path,
    parameterization: str,
    lr: float,
    run_name: str,
    out_dir: Path,
    data_dir: Path,
    max_tokens_override: int | None = None,
    max_steps_override: int | None = None,
    save_checkpoint: bool = False,
    warmstart_from: Path | None = None,
    dry_run: bool = False,
):
    cfg = load_yaml(config_path)
    arch = cfg["arch"]
    tcfg = cfg["train"]

    if max_tokens_override is not None:
        tcfg["max_tokens"] = int(max_tokens_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: running on CPU, this will be very slow")

    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    gen = np.random.default_rng(tcfg["seed"])

    out_dir = Path(out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_shapes_path = Path("results") / "mup_base_shapes.bsh"
    model = build_model(arch, parameterization, base_shapes_path).to(device)
    if warmstart_from is not None:
        warmstart_from = Path(warmstart_from)
        prior = torch.load(warmstart_from, map_location=device, weights_only=False)
        model.load_state_dict(prior["state_dict"])
        print(f"warmstarted from {warmstart_from}")
    n_params = model.num_params(include_embeddings=True)
    print(f"run={run_name} parameterization={parameterization} params={n_params:,} lr={lr:g}")

    optim = build_optimizer(model, parameterization, lr, tcfg["weight_decay"], (tcfg["beta1"], tcfg["beta2"]))

    block_size = arch["block_size"]
    micro_bs = tcfg["micro_batch_size"]
    micro_tokens = micro_bs * block_size
    grad_accum = max(1, tcfg["batch_tokens"] // micro_tokens)
    effective_batch_tokens = grad_accum * micro_tokens
    total_steps = tcfg["max_tokens"] // effective_batch_tokens
    if max_steps_override is not None:
        total_steps = min(total_steps, max_steps_override)
    warmup_steps = max(1, int(total_steps * tcfg["warmup_frac"]))
    eval_every_steps = max(1, tcfg["eval_interval_tokens"] // effective_batch_tokens)

    if dry_run:
        train_ds = None
        val_ds = None
    else:
        train_ds = TokenDataset(data_dir / "train.bin", block_size)
        val_ds = TokenDataset(data_dir / "val.bin", block_size)

    amp_dtype = autocast_dtype(tcfg["amp_dtype"])
    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "tokens", "lr", "train_loss", "val_loss", "tok_per_s", "gpu_mem_mb"])

    throughput = Throughput()
    best_val = float("inf")
    last_train = float("nan")
    t_start = time.time()

    for step in range(total_steps):
        lr_now = cosine_lr(step, total_steps, warmup_steps, lr)
        for g in optim.param_groups:
            g["lr"] = lr_now

        optim.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            if dry_run:
                x = torch.randint(0, arch["vocab_size"], (micro_bs, block_size), device=device)
                y = torch.randint(0, arch["vocab_size"], (micro_bs, block_size), device=device)
            else:
                x, y = train_ds.sample(micro_bs, device, gen)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_acc += loss.item()
            throughput.add(x.numel())
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optim.step()
        last_train = loss_acc / grad_accum

        do_eval = (step + 1) % eval_every_steps == 0 or (step + 1) == total_steps
        val_loss = ""
        if do_eval and not dry_run:
            val_loss_f = evaluate(model, val_ds, batch_size=min(32, micro_bs * 2), n_batches=20, device=device, amp_dtype=amp_dtype, gen=gen)
            val_loss = f"{val_loss_f:.6f}"
            best_val = min(best_val, val_loss_f)

        if (step + 1) % tcfg["log_interval_steps"] == 0 or do_eval:
            tokens_seen = (step + 1) * effective_batch_tokens
            mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step + 1, tokens_seen, f"{lr_now:.6g}", f"{last_train:.6f}", val_loss, f"{throughput.rate():.0f}", f"{mem:.0f}"])
            print(f"  step {step+1}/{total_steps} tok={tokens_seen:,} lr={lr_now:.4g} train={last_train:.4f} val={val_loss or '-'} tok/s={throughput.rate():,.0f}")

    final_val = float("nan") if dry_run else evaluate(model, val_ds, batch_size=min(32, micro_bs * 2), n_batches=50, device=device, amp_dtype=amp_dtype, gen=gen)
    wall = time.time() - t_start
    final = {
        "name": run_name,
        "config": str(config_path),
        "parameterization": parameterization,
        "lr": lr,
        "n_params": int(n_params),
        "tokens": int(total_steps * effective_batch_tokens),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "grad_accum": grad_accum,
        "effective_batch_tokens": effective_batch_tokens,
        "wall_seconds": wall,
        "tok_per_s": throughput.rate(),
        "max_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0,
        "val_loss_final": final_val,
        "val_loss_best": float(best_val),
        "train_loss_last": float(last_train),
    }
    with open(out_dir / "final.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2))

    if save_checkpoint:
        ckpt = {
            "state_dict": model.state_dict(),
            "arch": arch,
            "parameterization": parameterization,
            "base_shapes_path": str(base_shapes_path) if parameterization == "mup" else None,
        }
        torch.save(ckpt, out_dir / "model.pt")
        print(f"saved checkpoint to {out_dir / 'model.pt'}")

    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--parameterization", choices=["sp", "mup"], default="sp")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--save_checkpoint", action="store_true")
    p.add_argument("--warmstart_from", type=Path, default=None,
                   help="load model weights from this checkpoint, then train with a fresh schedule")
    p.add_argument("--dry_run", action="store_true",
                   help="use random tokens instead of real data; verifies model + GPU without needing data files")
    args = p.parse_args()
    train_one(
        config_path=args.config,
        parameterization=args.parameterization,
        lr=args.lr,
        run_name=args.run_name,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        max_tokens_override=args.max_tokens,
        max_steps_override=args.max_steps,
        save_checkpoint=args.save_checkpoint,
        warmstart_from=args.warmstart_from,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
