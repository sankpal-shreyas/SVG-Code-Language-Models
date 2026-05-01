"""Compute XML validity, render rate, structural validity. Optionally compute test perplexity.

Also produces the sample grid figure and prefix completions figure used in the report.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.normalize_svg import is_valid


SVG_OPEN = re.compile(r"<svg[\s>]")


def render(svg_str: str, out_png: Path, size: int = 192) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=str(out_png),
                         output_width=size, output_height=size)
        return out_png.exists() and out_png.stat().st_size > 0
    except Exception:
        pass
    try:
        from io import BytesIO
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(BytesIO(svg_str.encode("utf-8")))
        renderPM.drawToFile(drawing, str(out_png), fmt="PNG")
        return out_png.exists() and out_png.stat().st_size > 0
    except Exception:
        return False


def structural_check(svg_str: str) -> bool:
    if not SVG_OPEN.search(svg_str):
        return False
    if "</svg>" not in svg_str:
        return False
    return is_valid(svg_str)


def metrics_for_samples(samples_dir: Path, tmp_render: Path):
    files = sorted(samples_dir.glob("*.svg"))
    files = [f for f in files if not f.name.endswith(".prefix.svg")]
    n = len(files)
    if n == 0:
        return {"n": 0}
    valid_xml = 0
    structural = 0
    rendered = 0
    by_temp: dict[str, dict] = {}
    by_kind: dict[str, dict] = {}
    for i, f in enumerate(files):
        text = f.read_text(encoding="utf-8")
        v = is_valid(text)
        s = structural_check(text)
        r = render(text, tmp_render / f"{i}.png") if v else False
        valid_xml += int(v); structural += int(s); rendered += int(r)
        m = re.search(r"_T(\d\.\d)_", f.name)
        T = m.group(1) if m else "all"
        kind = "uncond" if f.name.startswith("uncond") else "prefix"
        for d, k in ((by_temp, T), (by_kind, kind)):
            d.setdefault(k, {"n": 0, "valid_xml": 0, "structural": 0, "rendered": 0})
            d[k]["n"] += 1
            d[k]["valid_xml"] += int(v)
            d[k]["structural"] += int(s)
            d[k]["rendered"] += int(r)
    return {
        "n": n,
        "valid_xml_rate": valid_xml / n,
        "structural_rate": structural / n,
        "render_rate": rendered / n,
        "by_temperature": by_temp,
        "by_kind": by_kind,
    }


def test_perplexity(ckpt_path: Path, data_bin: Path, block_size: int, n_batches: int = 200, batch_size: int = 16):
    import torch
    from model.gpt import GPTConfig, GPT
    from train.utils import TokenDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    if ckpt["parameterization"] == "mup":
        from model.mup_gpt import build_mup_model
        cfg = GPTConfig(**arch)
        model = build_mup_model(cfg, base_shapes_path=Path(ckpt["base_shapes_path"]) if ckpt.get("base_shapes_path") else None)
    else:
        cfg = GPTConfig(**arch)
        model = GPT(cfg)
    model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()

    ds = TokenDataset(data_bin, block_size)
    gen = np.random.default_rng(0)
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = ds.sample(batch_size, device, gen)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
    avg = float(np.mean(losses))
    return {"test_loss": avg, "test_ppl": math.exp(avg), "n_batches": n_batches}


def make_grid(samples_dir: Path, out_png: Path, rows: int = 4, cols: int = 4, size: int = 192):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    files = sorted(samples_dir.glob("uncond_T0.5_*.svg")) + sorted(samples_dir.glob("uncond_T0.8_*.svg"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = samples_dir / "_render"; tmp.mkdir(exist_ok=True)
    cells = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        if not is_valid(text):
            continue
        png = tmp / f"{f.stem}.png"
        if render(text, png, size=size):
            cells.append((png, f.stem))
        if len(cells) >= rows * cols:
            break

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (png, name) in zip(axes.flat, cells):
        ax.imshow(Image.open(png))
        ax.set_title(name, fontsize=6)
    fig.suptitle("unconditional samples (XML-valid only, T in {0.5, 0.8})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def make_prefix_panel(samples_dir: Path, out_png: Path, size: int = 192):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    pairs = []
    for f in sorted(samples_dir.glob("prefix_*_T0.8.svg")):
        if f.name.endswith(".prefix.svg"):
            continue
        prefix_file = f.with_name(f.name[:-len(".svg")] + ".prefix.svg")
        if not prefix_file.exists():
            continue
        pairs.append((prefix_file, f))

    if not pairs:
        print("no prefix completion pairs found")
        return
    rows = len(pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(4.5, rows * 2.0))
    if rows == 1:
        axes = np.array([axes])
    tmp = samples_dir / "_render"; tmp.mkdir(exist_ok=True)
    for r, (pf, cf) in enumerate(pairs):
        for c, fpath in enumerate((pf, cf)):
            text = fpath.read_text(encoding="utf-8")
            if not text.strip().endswith("</svg>"):
                text = text + "</svg>"
            png = tmp / f"{fpath.stem}_{c}.png"
            ok = render(text, png, size=size) if is_valid(text) else False
            ax = axes[r, c]
            ax.axis("off")
            if ok:
                ax.imshow(Image.open(png))
            ax.set_title(("prefix" if c == 0 else "completion") + " : " + pf.stem.replace("prefix_", "").replace(".prefix", ""),
                        fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, default=Path("results/best_samples"))
    p.add_argument("--out_metrics", type=Path, default=Path("results/validity_metrics.json"))
    p.add_argument("--grid_png", type=Path, default=Path("report/figures/sample_grid.png"))
    p.add_argument("--prefix_png", type=Path, default=Path("report/figures/prefix_completions.png"))
    p.add_argument("--ckpt", type=Path, default=None, help="optional, computes test perplexity")
    p.add_argument("--test_bin", type=Path, default=Path("data/test.bin"))
    p.add_argument("--block_size", type=int, default=1024)
    args = p.parse_args()

    tmp_render = args.samples / "_metric_render"
    tmp_render.mkdir(parents=True, exist_ok=True)
    metrics = metrics_for_samples(args.samples, tmp_render)
    if args.ckpt is not None and args.ckpt.exists():
        metrics["test"] = test_perplexity(args.ckpt, args.test_bin, args.block_size)

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2))
    make_grid(args.samples, args.grid_png)
    make_prefix_panel(args.samples, args.prefix_png)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
