"""Master data pipeline.

Stages:
  ingest    download + normalize SVGs from HF datasets, split 98/1/1 into JSONL caches
  tokenize  train BPE tokenizer on train split (skipped if tokenizer.json exists)
  pack      tokenize all splits, drop sequences > max_seq_tokens, write .bin files

Run all stages: python -m data.prepare
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.normalize_svg import normalize  # noqa: E402

DATA_DIR = Path("data")
CACHE = DATA_DIR / "cache"

SVG_FIELD_CANDIDATES = ("Svg", "svg", "code", "svg_code", "text", "Code", "data")


def find_svg_field(example: dict) -> str | None:
    for k in SVG_FIELD_CANDIDATES:
        v = example.get(k)
        if isinstance(v, str) and "<svg" in v:
            return v
    for v in example.values():
        if isinstance(v, str) and len(v) > 50 and "<svg" in v:
            return v
    return None


def split_for(name: str, idx: int, seed: int = 1337) -> str:
    h = hashlib.md5(f"{name}|{idx}|{seed}".encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < 0.98:
        return "train"
    if bucket < 0.99:
        return "val"
    return "test"


def ingest(target_train_chars: float, seed: int = 1337):
    from datasets import load_dataset

    CACHE.mkdir(parents=True, exist_ok=True)
    files = {s: open(CACHE / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    counts = {s: 0 for s in files}
    chars = {s: 0 for s in files}
    rejected = 0

    sources = [
        ("starvector/svg-icons-simple", False),
        ("starvector/svg-emoji-simple", False),
        ("starvector/svg-fonts-simple", True),
    ]
    try:
        for name, stream in sources:
            if chars["train"] >= target_train_chars:
                break
            print(f"\nloading {name} (streaming={stream})")
            try:
                ds = load_dataset(name, split="train", streaming=stream)
            except Exception as e:
                print(f"  skipping {name}: {e}")
                continue
            for idx, ex in enumerate(tqdm(ds, desc=name)):
                raw = find_svg_field(ex)
                if raw is None:
                    rejected += 1
                    continue
                cleaned = normalize(raw)
                if cleaned is None or len(cleaned) < 50:
                    rejected += 1
                    continue
                s = split_for(name, idx, seed)
                files[s].write(json.dumps({"svg": cleaned}, ensure_ascii=False) + "\n")
                counts[s] += 1
                chars[s] += len(cleaned)
                if stream and chars["train"] >= target_train_chars:
                    print(f"  hit char budget at idx={idx}, stopping")
                    break
    finally:
        for f in files.values():
            f.close()

    summary = {
        "files": counts,
        "chars": chars,
        "rejected": rejected,
        "target_train_chars": target_train_chars,
    }
    (CACHE / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfiles: {counts}\nchars: {chars}\nrejected: {rejected}")


def pack(tokenizer_path: Path, max_seq_tokens: int = 2048, batch_size: int = 1024):
    from data.train_tokenizer import load_tokenizer, eot_id

    tok = load_tokenizer(tokenizer_path)
    eot = eot_id(tok)
    if eot is None:
        raise RuntimeError("tokenizer is missing the <|endoftext|> special token")

    summary = {}
    for split in ("train", "val", "test"):
        in_path = CACHE / f"{split}.jsonl"
        out_path = DATA_DIR / f"{split}.bin"
        if not in_path.exists():
            print(f"skip {split}: {in_path} missing")
            continue
        chunks: list[np.ndarray] = []
        kept = 0
        dropped_long = 0
        total_lines = sum(1 for _ in open(in_path, "r", encoding="utf-8"))
        with open(in_path, "r", encoding="utf-8") as f:
            buf: list[str] = []
            with tqdm(total=total_lines, desc=f"pack {split}") as pbar:
                for line in f:
                    buf.append(json.loads(line)["svg"])
                    if len(buf) >= batch_size:
                        encs = tok.encode_batch(buf)
                        for e in encs:
                            ids = e.ids
                            if len(ids) > max_seq_tokens:
                                dropped_long += 1
                                continue
                            chunks.append(np.fromiter(ids, dtype=np.uint16))
                            chunks.append(np.array([eot], dtype=np.uint16))
                            kept += 1
                        pbar.update(len(buf))
                        buf.clear()
                if buf:
                    encs = tok.encode_batch(buf)
                    for e in encs:
                        ids = e.ids
                        if len(ids) > max_seq_tokens:
                            dropped_long += 1
                            continue
                        chunks.append(np.fromiter(ids, dtype=np.uint16))
                        chunks.append(np.array([eot], dtype=np.uint16))
                        kept += 1
                    pbar.update(len(buf))

        arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)
        arr.tofile(out_path)
        print(f"  {split}: {kept:,} kept, {dropped_long:,} dropped, {arr.size:,} tokens -> {out_path}")
        summary[split] = {"kept": kept, "dropped_long": dropped_long, "tokens": int(arr.size)}

    (DATA_DIR / "pack_summary.json").write_text(json.dumps(summary, indent=2))
    train_tokens = summary.get("train", {}).get("tokens", 0)
    if train_tokens < 100_000_000:
        print(f"\nWARNING: train tokens = {train_tokens:,} which is below the 100M target.")
        print("Re-run with a larger --target_train_chars or pull from svg-stack-simple.")
    else:
        print(f"\nOK: train tokens = {train_tokens:,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["ingest", "tokenize", "pack", "all"], default="all")
    p.add_argument("--target_train_chars", type=float, default=4.0e8)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--max_seq_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if args.stage in ("ingest", "all"):
        ingest(args.target_train_chars, args.seed)

    tok_path = DATA_DIR / "tokenizer.json"
    if args.stage in ("tokenize", "all") and not tok_path.exists():
        from data.train_tokenizer import train_tokenizer
        train_tokenizer(CACHE / "train.jsonl", tok_path, args.vocab_size)

    if args.stage in ("pack", "all"):
        pack(tok_path, args.max_seq_tokens)


if __name__ == "__main__":
    main()
