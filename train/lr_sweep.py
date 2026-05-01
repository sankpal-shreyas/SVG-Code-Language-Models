"""Run a sequential LR sweep on a small model and pick the best."""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train.train import train_one
from train.utils import load_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_config", type=Path)
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    args = p.parse_args()

    sweep = load_yaml(args.sweep_config)
    base_cfg = Path(sweep["base_config"])
    parameterization = sweep["parameterization"]
    lrs = [float(x) for x in sweep["learning_rates"]]
    sweep_max_tokens = int(sweep["sweep_max_tokens"])
    out_path = Path(sweep.get("output", f"results/{parameterization}_best_lr.json"))

    results = []
    for lr in lrs:
        run_name = f"{parameterization}_lrsweep_lr{lr:g}"
        print(f"\n=== sweep run {run_name} ===")
        final = train_one(
            config_path=base_cfg,
            parameterization=parameterization,
            lr=lr,
            run_name=run_name,
            out_dir=args.out_dir,
            data_dir=args.data_dir,
            max_tokens_override=sweep_max_tokens,
        )
        results.append({"lr": lr, "val_loss": final["val_loss_final"], "best_val": final["val_loss_best"]})

    results.sort(key=lambda r: r["val_loss"])
    best = results[0]
    summary = {
        "parameterization": parameterization,
        "best_lr": best["lr"],
        "best_val_loss": best["val_loss"],
        "all": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nbest LR for {parameterization}: {best['lr']:g}  (val={best['val_loss']:.4f})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
