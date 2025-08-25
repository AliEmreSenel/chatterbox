"""Extend tokenizer with Turkish tokens and save a new tokenizer.json

This script expects a HF-style tokenizer.json present in ckpt_dir/tokenizer.json.
It will load that tokenizer via tokenizers.Tokenizer and add new tokens from the dataset text.
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


def collect_unique_words(metadata_tsv: Path, limit: int = None):
    words = set()
    with open(metadata_tsv, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            path, text = line.strip().split("\t", 1)
            for w in text.split():
                words.add(w)
    return words


def main(ckpt_dir: str, data_meta: str, out_dir: str):
    ck = Path(ckpt_dir)
    tok = Tokenizer.from_file(str(ck / "tokenizer.json"))
    words = collect_unique_words(Path(data_meta))
    # filter simple punctuation and lower-case
    words = {w.strip().lower() for w in words if w.strip()}
    new_tokens = []
    for w in words:
        # skip tokens already in vocab
        if w in tok.get_vocab():
            continue
        if len(w) <= 3:
            continue
        new_tokens.append(w)
    if not new_tokens:
        print("no new tokens")
        return
    tok.add_tokens(new_tokens)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer_tr.json"))
    print(f'saved tokenizer to {out / "tokenizer_tr.json"}')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="ckpt")
    p.add_argument("--data_meta", default="data/common_voice_tr/metadata.tsv")
    p.add_argument("--out_dir", default="ckpt_tr")
    args = p.parse_args()
    main(args.ckpt_dir, args.data_meta, args.out_dir)
