"""Plot val loss vs LR for SP and muP sweeps on the same axes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(results_dir: Path, prefix: str):
    rows = []
    for d in sorted(results_dir.glob(f"{prefix}_lrsweep_lr*")):
        f = d / "final.json"
        if not f.exists():
            continue
        rows.append(json.loads(f.read_text()))
    rows.sort(key=lambda r: r["lr"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("report/figures/lr_sweep.png"))
    args = p.parse_args()

    sp = collect(args.results_dir, "sp")
    mup = collect(args.results_dir, "mup")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if sp:
        sx = [r["lr"] for r in sp]; sy = [r["val_loss_final"] for r in sp]
        ax.plot(sx, sy, marker="o", color="#1f77b4", label="SP")
        i = min(range(len(sy)), key=lambda j: sy[j])
        ax.scatter([sx[i]], [sy[i]], facecolors="none", edgecolors="#1f77b4", s=180, linewidths=2, zorder=5)
    if mup:
        mx = [r["lr"] for r in mup]; my = [r["val_loss_final"] for r in mup]
        ax.plot(mx, my, marker="s", color="#d62728", label=r"$\mu$P")
        i = min(range(len(my)), key=lambda j: my[j])
        ax.scatter([mx[i]], [my[i]], facecolors="none", edgecolors="#d62728", s=180, linewidths=2, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("validation loss after 10M tokens (Tiny)")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
