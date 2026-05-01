"""Dataset statistics, length histogram, and rendered icon examples."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))


def render_svg(svg_str: str, out_png: Path, size: int = 128) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=str(out_png),
                         output_width=size, output_height=size)
        return True
    except Exception:
        pass
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        from io import BytesIO
        drawing = svg2rlg(BytesIO(svg_str.encode("utf-8")))
        renderPM.drawToFile(drawing, str(out_png), fmt="PNG")
        return True
    except Exception as e:
        print(f"render failed for {out_png.name}: {e}")
        return False


def length_histogram(jsonl_path: Path, tokenizer_path: Path, out_dir: Path):
    from data.train_tokenizer import load_tokenizer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tok = load_tokenizer(tokenizer_path)
    lens = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="len"):
            s = json.loads(line)["svg"]
            lens.append(len(tok.encode(s).ids))
    lens = np.array(lens)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lens, bins=80, range=(0, 2048), color="#444", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("tokens per SVG (BPE, vocab 4096)")
    ax.set_ylabel("count")
    ax.set_title(f"sequence length distribution, train (n={len(lens):,})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "seq_len_hist.png", dpi=150)
    plt.close(fig)

    return {
        "n": int(len(lens)),
        "mean": float(lens.mean()),
        "median": float(np.median(lens)),
        "p95": float(np.percentile(lens, 95)),
        "p99": float(np.percentile(lens, 99)),
        "max": int(lens.max()),
    }


def render_examples(jsonl_path: Path, tokenizer_path: Path, out_dir: Path, n: int = 6):
    from data.train_tokenizer import load_tokenizer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    tok = load_tokenizer(tokenizer_path)
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)["svg"]
            items.append((len(tok.encode(s).ids), s))
            if len(items) >= 5000:
                break
    items.sort(key=lambda x: x[0])
    indices = np.linspace(0, len(items) - 1, n).astype(int)
    chosen = [items[i] for i in indices]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    for ax, (toks, svg) in zip(axes, chosen):
        png = out_dir / f"_ex_{toks}.png"
        ok = render_svg(svg, png, size=192)
        if ok:
            img = Image.open(png)
            ax.imshow(img)
        ax.set_title(f"{toks} tok", fontsize=9)
        ax.axis("off")
    fig.suptitle("example icons across the length distribution", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "example_icons.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--bins", type=Path, default=Path("data"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    p.add_argument("--figs", type=Path, default=Path("report/figures"))
    p.add_argument("--out", type=Path, default=Path("results/dataset_stats.json"))
    args = p.parse_args()

    summary = {}
    for split in ("train", "val", "test"):
        path = args.bins / f"{split}.bin"
        if path.exists():
            arr = np.fromfile(path, dtype=np.uint16)
            summary[f"{split}_tokens"] = int(arr.size)

    pack_summary = args.bins / "pack_summary.json"
    if pack_summary.exists():
        summary["pack"] = json.loads(pack_summary.read_text())

    train_jsonl = args.cache / "train.jsonl"
    if train_jsonl.exists() and args.tokenizer.exists():
        summary["lengths"] = length_histogram(train_jsonl, args.tokenizer, args.figs)
        render_examples(train_jsonl, args.tokenizer, args.figs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
