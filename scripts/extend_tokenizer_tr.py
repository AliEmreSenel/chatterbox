"""
Extend an existing English tokenizer with Turkish subwords
while keeping original English token IDs.

- Loads tokenizer.json from ckpt_dir.
- Trains a new tokenizer only on Turkish text.
- Adds new subwords not already in English vocab.
- Appends them to the original tokenizer.
- Saves tokenizer_en_tr.json.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer


def train_turkish_tokenizer(corpus_file: Path, vocab_size: int = 8000):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        initial_alphabet=list("abcçdefgğhıijklmnoöprsştuüvyz"),
    )
    tok.train([str(corpus_file)], trainer)
    return tok


def extend_with_turkish(base_tok: Tokenizer, turkish_tok: Tokenizer, out_path: Path):
    """Append Turkish tokens to base tokenizer vocab while preserving English IDs."""
    base_vocab = set(base_tok.get_vocab().keys())
    turkish_vocab = list(turkish_tok.get_vocab().keys())

    # filter out tokens already present in English
    new_tokens = [tok for tok in turkish_vocab if tok not in base_vocab]

    if not new_tokens:
        print("⚠️ No new Turkish tokens to add.")
        return

    # Append new tokens (IDs will be added after English ones)
    base_tok.add_tokens(new_tokens)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_tok.save(str(out_path))
    print(f"✅ Extended tokenizer saved to: {out_path}")
    print(f"➕ Added {len(new_tokens)} new tokens.")


def main(ckpt_dir: str, tr_tsv: str, out_dir: str, turkish_vocab: int = 8000):
    # Load English tokenizer
    base_tok = Tokenizer.from_file(str(Path(ckpt_dir) / "tokenizer.json"))

    # Train a Turkish-only tokenizer
    turkish_tok = train_turkish_tokenizer(Path(tr_tsv), vocab_size=turkish_vocab)

    # Extend English tokenizer with Turkish tokens
    out_path = Path(out_dir) / "tokenizer_en_tr.json"
    extend_with_turkish(base_tok, turkish_tok, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extend tokenizer with Turkish while preserving English IDs"
    )
    parser.add_argument(
        "--ckpt_dir", default="ckpt", help="Directory with English tokenizer.json"
    )
    parser.add_argument(
        "--tr_tsv",
        default="data/common_voice_tr/metadata.tsv",
        help="TSV with Turkish text",
    )
    parser.add_argument("--out_dir", default="ckpt_tr", help="Output directory")
    parser.add_argument(
        "--turkish_vocab", type=int, default=704, help="Size of Turkish-only vocab"
    )
    args = parser.parse_args()

    main(args.ckpt_dir, args.tr_tsv, args.out_dir, args.turkish_vocab)
