"""Plot validation loss vs tokens for SP and muP scaling runs, one panel per size."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIZES = ["tiny", "small", "medium", "large", "xl"]
SIZE_LABEL = {"tiny": "Tiny", "small": "Small", "medium": "Medium", "large": "Large", "xl": "XL"}


def read_val(run_dir: Path):
    f = run_dir / "log.csv"
    if not f.exists():
        return [], []
    xs, ys = [], []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if row.get("val_loss"):
                xs.append(float(row["tokens"]))
                ys.append(float(row["val_loss"]))
    return xs, ys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("report/figures/val_curves.png"))
    args = p.parse_args()

    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.2), sharey=False)
    for ax, sz in zip(axes, SIZES):
        sp_x, sp_y = read_val(args.results_dir / f"sp_{sz}")
        mup_x, mup_y = read_val(args.results_dir / f"mup_{sz}")
        if sp_x:
            ax.plot(sp_x, sp_y, color="#1f77b4", label="SP", linewidth=1.6)
        if mup_x:
            ax.plot(mup_x, mup_y, color="#d62728", label=r"$\mu$P", linewidth=1.6)
        ax.set_xscale("log")
        ax.set_title(SIZE_LABEL[sz], fontsize=10)
        ax.set_xlabel("tokens seen")
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
        if sz == "tiny":
            ax.set_ylabel("validation loss")
            ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
