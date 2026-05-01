"""Fit power laws to SP and muP scaling runs, plot, and write extrapolation."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.append(str(Path(__file__).resolve().parents[1]))


def power_law(N, a, alpha, c):
    return a * np.power(N, -alpha) + c


def collect(run_dirs):
    rows = []
    for d in run_dirs:
        f = Path(d) / "final.json"
        if not f.exists():
            print(f"  missing: {f}")
            continue
        rows.append(json.loads(f.read_text()))
    rows.sort(key=lambda r: r["n_params"])
    return rows


def fit(rows, n_first: int | None = None):
    """Fit log(L) = log(a) - alpha*log(N) by ordinary least squares in log-log space.
    This is the 2-parameter Kaplan-style power law (no irreducible-loss offset).
    With three points it yields 1 degree of freedom, enough for meaningful std errors,
    where the 3-parameter form L = a*N^(-alpha) + c is exactly determined and underconstrained."""
    if n_first is not None:
        rows = rows[:n_first]
    N = np.array([r["n_params"] for r in rows], dtype=np.float64)
    L = np.array([r["val_loss_final"] for r in rows], dtype=np.float64)
    if len(N) < 3:
        print(f"  too few points to fit ({len(N)})")
        return None
    x = np.log(N); y = np.log(L)
    n = len(x)
    A = np.vstack([np.ones(n), -x]).T  # y = log_a - alpha * x
    beta, residuals, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    log_a, alpha = beta
    yhat = A @ beta
    sse = float(np.sum((y - yhat) ** 2))
    dof = max(n - 2, 1)
    sigma2 = sse / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    log_a_err = float(np.sqrt(cov[0, 0]))
    alpha_err = float(np.sqrt(cov[1, 1]))
    a = float(np.exp(log_a))
    return {"a": a, "alpha": float(alpha), "c": 0.0,
            "a_err": float(a * log_a_err), "alpha_err": alpha_err, "c_err": 0.0,
            "N": N.tolist(), "L": L.tolist(),
            "n_points": int(n),
            "rmse": float(np.sqrt(sigma2))}


def predict_with_ci(fit_res, N_query, n_samples=2000):
    if fit_res is None:
        return None
    rng = np.random.default_rng(0)
    log_a = np.log(max(fit_res["a"], 1e-12))
    log_a_err = fit_res["a_err"] / max(fit_res["a"], 1e-12)
    samples = []
    for _ in range(n_samples):
        la = rng.normal(log_a, log_a_err)
        alpha = rng.normal(fit_res["alpha"], fit_res["alpha_err"])
        samples.append(np.exp(la) * np.power(N_query, -alpha) + fit_res["c"])
    samples = np.array(samples)
    return {
        "N_query": float(N_query),
        "mean": float(power_law(N_query, fit_res["a"], fit_res["alpha"], fit_res["c"])),
        "p2_5": float(np.percentile(samples, 2.5)),
        "p97_5": float(np.percentile(samples, 97.5)),
    }


def plot(fits: dict, all_points: dict, out_path: Path):
    """Plot all data points, plus the fitted power law on the points used for the fit.
    Excluded points (LR miscalibrated) are drawn as open markers and omitted from the fit line."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"sp": "#1f77b4", "mup": "#d62728"}
    for name, fr in fits.items():
        rows = all_points.get(name) or []
        if not rows:
            continue
        N_all = np.array([r["n_params"] for r in rows], dtype=np.float64)
        L_all = np.array([r["val_loss_final"] for r in rows], dtype=np.float64)
        n_used = fr["n_points"] if fr else 0
        ax.scatter(N_all[:n_used], L_all[:n_used], color=colors.get(name, "k"),
                   label=f"{name.upper()} (in fit)", zorder=3)
        if n_used < len(N_all):
            ax.scatter(N_all[n_used:], L_all[n_used:], facecolors="none",
                       edgecolors=colors.get(name, "k"),
                       label=f"{name.upper()} (excluded, LR miscalibrated)", zorder=3)
        if fr is not None:
            Nfit = np.geomspace(N_all.min() * 0.8, N_all.max() * 12.0, 200)
            Lfit = power_law(Nfit, fr["a"], fr["alpha"], fr["c"])
            ax.plot(Nfit, Lfit, color=colors.get(name, "k"), linestyle="--",
                    label=f"{name.upper()} fit (n={n_used}): alpha={fr['alpha']:.3f}+/-{fr['alpha_err']:.3f}, c={fr['c']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("validation loss after 1 epoch")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def monotone_prefix(rows):
    """Return the longest leading prefix of rows where val_loss is non-increasing in n_params."""
    if not rows:
        return rows
    keep = [rows[0]]
    for r in rows[1:]:
        if r["val_loss_final"] <= keep[-1]["val_loss_final"] + 1e-3:
            keep.append(r)
        else:
            break
    return keep


def plot_curves(run_dirs, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for d in run_dirs:
        f = Path(d) / "log.csv"
        if not f.exists():
            continue
        toks, train, val_x, val_y = [], [], [], []
        with open(f, "r") as fh:
            r = csv.DictReader(fh)
            for row in r:
                t = float(row["tokens"])
                toks.append(t); train.append(float(row["train_loss"]))
                if row["val_loss"]:
                    val_x.append(t); val_y.append(float(row["val_loss"]))
        ax.plot(toks, train, alpha=0.6, label=Path(d).name)
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("train loss")
    ax.set_xscale("log")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sp_runs", nargs="*", default=[])
    p.add_argument("--mup_runs", nargs="*", default=[])
    p.add_argument("--out_plot", type=Path, default=Path("report/figures/sp_vs_mup_scaling.png"))
    p.add_argument("--out_summary", type=Path, default=Path("results/scaling_summary.json"))
    p.add_argument("--curves_plot", type=Path, default=Path("report/figures/training_curves.png"))
    p.add_argument("--predict_x", type=float, default=10.0, help="extrapolate at predict_x times the largest N")
    args = p.parse_args()

    sp_rows = collect(args.sp_runs)
    mup_rows = collect(args.mup_runs)

    # Fit only the monotonic prefix. Points that break monotonicity are typically
    # cases where the proxy-tuned LR was miscalibrated for that width, and including
    # them yields a nonsense power law.
    sp_used = monotone_prefix(sp_rows)
    mup_used = monotone_prefix(mup_rows)
    fits = {"sp": fit(sp_used) if len(sp_used) >= 3 else None,
            "mup": fit(mup_used) if len(mup_used) >= 3 else None}

    summary = {"sp": fits["sp"], "mup": fits["mup"],
               "sp_excluded": [r["name"] for r in sp_rows[len(sp_used):]],
               "mup_excluded": [r["name"] for r in mup_rows[len(mup_used):]]}

    largest_N = 0
    for rows in (sp_rows, mup_rows):
        if rows:
            largest_N = max(largest_N, rows[-1]["n_params"])
    if largest_N > 0:
        N_query = args.predict_x * largest_N
        summary["extrapolation"] = {
            "N_query": N_query,
            "factor_over_largest": args.predict_x,
            "sp": predict_with_ci(fits["sp"], N_query),
            "mup": predict_with_ci(fits["mup"], N_query),
        }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2))
    plot(fits, {"sp": sp_rows, "mup": mup_rows}, args.out_plot)
    plot_curves(args.sp_runs + args.mup_runs, args.curves_plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
