"""Train a byte-level BPE tokenizer on a sample of normalized SVGs."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


END_OF_TEXT = "<|endoftext|>"


def train_tokenizer(jsonl_path: Path, out_path: Path, vocab_size: int = 4096, max_chars: int = 10_000_000):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[END_OF_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def iter_corpus():
        chars = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                s = row["svg"]
                yield s
                chars += len(s)
                if chars >= max_chars:
                    return

    tok.train_from_iterator(iter_corpus(), trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f"saved tokenizer with vocab={tok.get_vocab_size()} to {out_path}")


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def eot_id(tok: Tokenizer) -> int:
    return tok.token_to_id(END_OF_TEXT)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("data/cache/train.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/tokenizer.json"))
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--max_chars", type=int, default=10_000_000)
    args = p.parse_args()
    train_tokenizer(args.corpus, args.out, args.vocab_size, args.max_chars)
