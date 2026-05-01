"""Sample SVGs from a trained checkpoint, unconditional and prefix-conditioned."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.gpt import GPT, GPTConfig
from data.train_tokenizer import load_tokenizer, eot_id
from data.normalize_svg import is_valid


PREFIXES = {
    "uncond": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">',
    "partial_face": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="#000"/><circle cx="9" cy="10" r="1" fill="#000"/>',
    "open_path": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M4 4 L20 4 L20 20',
    "single_shape_group": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><g><rect x="2" y="2" width="8" height="8" fill="#444"/>',
    "half_stroked_rect": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" stroke="#000" stroke-width="2" fill="none"',
    "mirror_half": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M2 12 L12 2 L12 22 Z" fill="#888"/>',
}


def load_checkpoint(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    parameterization = ckpt["parameterization"]
    base_shapes_path = ckpt.get("base_shapes_path")
    if parameterization == "mup":
        from model.mup_gpt import build_mup_model
        cfg = GPTConfig(**arch)
        model = build_mup_model(cfg, base_shapes_path=Path(base_shapes_path) if base_shapes_path else None)
    else:
        cfg = GPTConfig(**arch)
        model = GPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def sample(model, tok, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, device, eos):
    ids = tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, eos_id=eos)
    out_ids = out[0].tolist()
    if eos in out_ids[len(ids):]:
        cut = out_ids.index(eos, len(ids))
        out_ids = out_ids[:cut]
    return tok.decode(out_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    p.add_argument("--out_dir", type=Path, default=Path("results/best_samples"))
    p.add_argument("--n_uncond", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.ckpt, device)
    tok = load_tokenizer(args.tokenizer)
    eos = eot_id(tok)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for T in args.temperatures:
        for i in range(args.n_uncond):
            text = sample(model, tok, PREFIXES["uncond"], args.max_new_tokens, T, args.top_p, args.top_k, device, eos)
            name = f"uncond_T{T:.1f}_i{i:02d}"
            (args.out_dir / f"{name}.svg").write_text(text, encoding="utf-8")
            manifest.append({"name": name, "kind": "uncond", "temperature": T,
                             "valid_xml": is_valid(text), "len_chars": len(text)})

        for pname, prefix in PREFIXES.items():
            if pname == "uncond":
                continue
            text = sample(model, tok, prefix, args.max_new_tokens, T, args.top_p, args.top_k, device, eos)
            name = f"prefix_{pname}_T{T:.1f}"
            (args.out_dir / f"{name}.svg").write_text(text, encoding="utf-8")
            (args.out_dir / f"{name}.prefix.svg").write_text(prefix, encoding="utf-8")
            manifest.append({"name": name, "kind": "prefix", "prefix_name": pname,
                             "temperature": T, "valid_xml": is_valid(text), "len_chars": len(text)})

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    valid = sum(1 for m in manifest if m["valid_xml"])
    print(f"wrote {len(manifest)} samples to {args.out_dir}, {valid}/{len(manifest)} parse as valid XML")


if __name__ == "__main__":
    main()
